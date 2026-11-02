"""Size-pinned, resumable selective acquisition followed by CPU E0 preprocessing."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlparse
import zipfile
import zlib

from e0.acquisition import RangeFile, validate_range_headers
from e0.data import check_leakage, load_actions, load_segments, sha256
from e0.preprocess import prepare
from e0.quality import EXCLUDED_RECORDINGS
from e0.source_profile import dimensionless_profile
from scripts.data.inspect_e0_sources import REVISION

EXPECTED_TOTALS = (7994397395, 23098116832)


def local_member(root, member):
    parts = Path(member["path"]).parts
    if len(parts) != 3 or parts[0] != "assembly101_camera_and_hand_poses" or parts[1] not in ("landmarks3D", "timestamp", "hand_confidences"):
        raise ValueError("Unexpected planned member path")
    path = root / parts[1] / parts[2]
    if not path.resolve().is_relative_to(root.resolve()) or Path(parts[2]).stem in EXCLUDED_RECORDINGS:
        raise ValueError("Unsafe or excluded member path")
    return path


def verify_file(path, size, crc):
    if not path.is_file() or path.stat().st_size != size:
        return None
    digest, checksum = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            checksum = zlib.crc32(block, checksum)
    return digest.hexdigest() if checksum == crc else None


def acquire_member(archive, member, destination):
    """Read in large blocks to avoid one remote request per small JSON fragment."""
    info = archive.getinfo(member["path"])
    if (info.compress_size, info.file_size) != (member["compressed_bytes"], member["uncompressed_bytes"]):
        raise ValueError("Planned ZIP member size changed")
    existing = verify_file(destination, info.file_size, info.CRC)
    if existing:
        return {"path": str(destination), "sha256": existing, "bytes": info.file_size, "zip_crc": info.CRC, "reused": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".json.part")
    if partial.is_symlink():
        raise ValueError("Refuse symlink partial file")
    digest, size = hashlib.sha256(), 0
    with archive.open(info) as source, partial.open("wb") as target:
        while block := source.read(16 * 1024 * 1024):
            size += len(block)
            if size > info.file_size:
                raise ValueError("Expanded member exceeded planned size")
            digest.update(block)
            target.write(block)
    if size != info.file_size:
        raise ValueError("Truncated ZIP member")
    # ZipExtFile verifies the member CRC when reaching EOF, before publication.
    os.replace(partial, destination)
    return {"path": str(destination), "sha256": digest.hexdigest(), "bytes": size, "zip_crc": info.CRC, "reused": False}


def assemble_manifest(plan, review, acquired):
    files = {e["name"]: e for e in review["files"]}
    actions = load_actions(files["actions.csv"]["path"])
    segments, _ = load_segments({s: files[f"{s}.csv"]["path"] for s in ("train", "validation", "test")}, actions)
    check_leakage(segments, "report_official_overlap")
    membership = {s["recording_id"]: {"split": s["split"], "participant_id": s["participant_id"]} for s in segments}
    grouped = {}
    for member in plan["members"]:
        kind, rec = Path(member["path"]).parent.name, Path(member["path"]).stem
        if rec not in membership:
            raise ValueError("Acquired member has no official split membership")
        entry = grouped.setdefault(rec, dict(membership[rec]))
        value = acquired[member["path"]]
        source = {"path": value["path"], "sha256": value["sha256"]}
        if kind == "landmarks3D":
            entry.update(source)
        else:
            entry["confidence" if kind == "hand_confidences" else "timestamp"] = source
    if any(not {"path", "sha256", "confidence", "timestamp"}.issubset(e) for e in grouped.values()):
        raise ValueError("Incomplete recording acquisition")
    return {"status": "validated", "dataset_revision": REVISION,
            "actions": {k: files["actions.csv"][k] for k in ("path", "sha256")},
            "splits": {s: {k: files[f"{s}.csv"][k] for k in ("path", "sha256")} for s in ("train", "validation", "test")},
            "recordings": grouped, "pose_profile": dimensionless_profile(),
            "protocol_scope": "curated pose-available subset; official split membership; reference-reader timing",
            "plan_sha256": plan["plan_sha256"],
            "selected_action_ids": [c["action_id"] for c in plan["class_mapping"]]}


def main():
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction") or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dedicated project CPU Slurm execution required")
    if not Path(os.environ.get("HF_HOME", "/nonexistent")).resolve().is_relative_to(root):
        raise RuntimeError("Dedicated HF cache required")
    plan_path = root / "data/annotations/e0-acquisition-plan-242776.json"
    review_path = root / "data/annotations/e0-review-242218/inspection.json"
    plan, review = (json.loads(p.read_text()) for p in (plan_path, review_path))
    if plan["revision"] != REVISION or review["revision"] != REVISION or plan["review_sha256"] != sha256(review_path) or "failure" in plan:
        raise RuntimeError("Source plan/review provenance mismatch")
    totals = (sum(m["compressed_bytes"] for m in plan["members"]), sum(m["uncompressed_bytes"] for m in plan["members"]))
    if totals != EXPECTED_TOTALS or len(plan["members"]) != 1005 or len(plan["class_mapping"]) != 100:
        raise RuntimeError("Reported 335-recording acquisition plan changed")
    for entry in review["files"]:
        if sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Official annotation hash changed")
    raw = root / "data/raw/assembly101-e0"
    if not raw.resolve().is_relative_to(root):
        raise RuntimeError("Raw data directory must stay inside the dedicated project")
    destinations = {m["path"]: local_member(raw, m) for m in plan["members"]}
    if len(destinations) != len(plan["members"]):
        raise RuntimeError("Duplicate planned member")
    missing_bytes = sum(m["uncompressed_bytes"] for m in plan["members"] if not destinations[m["path"]].is_file()
                        or destinations[m["path"]].stat().st_size != m["uncompressed_bytes"])
    used = int(subprocess.check_output(["du", "-s", "-B1", str(root)], text=True).split()[0])
    feature_bound = sum(row["segments_before_quality_checks"] for row in plan["coverage"].values()) * 64 * 216 * 4
    reserve = missing_bytes + feature_bound + 3000000000
    if used + reserve > 50000000000 or shutil.disk_usage(root).free < reserve:
        raise RuntimeError("Insufficient headroom under dedicated 50 GB budget; do not delete other project data")
    output = root / f"data/annotations/e0-preparation-{os.environ['SLURM_JOB_ID']}"
    output.mkdir(exist_ok=False)
    plan["plan_sha256"] = sha256(plan_path)
    report = {"plan_sha256": plan["plan_sha256"], "initial_project_allocated_bytes": used,
              "planned_extracted_bytes": totals[1], "planned_compressed_bytes": totals[0],
              "feature_bytes_upper_bound": feature_bound, "training_ready": False, "acquired": {}}
    print(json.dumps({k: v for k, v in report.items() if k != "acquired"}), flush=True)
    stage = "acquisition"
    try:
        from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
        import httpx
        token = get_token()
        if not token:
            raise RuntimeError("Private HF authentication required")
        url = hf_hub_url("cvml-nus/assembly101", "AssemblyPoses.zip", repo_type="dataset", revision=REVISION)
        metadata = get_hf_file_metadata(url, token=token)
        if metadata.size != 71972006749:
            raise RuntimeError("Pinned archive size changed")
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            def fetch(start, count):
                nonlocal metadata
                data = bytearray()
                while len(data) < count:
                    offset = start + len(data)
                    length = min(2000000, count - len(data))
                    for attempt in range(2):
                        auth = {"Authorization": f"Bearer {token}"} if urlparse(metadata.location).hostname == "huggingface.co" else {}
                        headers = {**auth, "Range": f"bytes={offset}-{offset + length - 1}", "Accept-Encoding": "identity"}
                        with client.stream("GET", metadata.location, headers=headers) as response:
                            if response.status_code == 403 and attempt == 0:
                                metadata = get_hf_file_metadata(url, token=token)
                                if metadata.size != 71972006749:
                                    raise RuntimeError("Archive size changed on URL renewal")
                                continue
                            validate_range_headers(response.status_code, response.headers.get("Content-Range"), offset, length, metadata.size)
                            block_bytes = bytearray()
                            for block in response.iter_bytes(chunk_size=65536):
                                if len(block_bytes) + len(block) > length:
                                    raise RuntimeError("Range body exceeded bound")
                                block_bytes.extend(block)
                            if len(block_bytes) != length:
                                raise RuntimeError("Truncated range")
                            data.extend(block_bytes)
                            break
                return bytes(data)
            stream = RangeFile(metadata.size, fetch, max_request=32000000, max_total=8100000000)
            try:
                with zipfile.ZipFile(stream) as archive:
                    for n, member in enumerate(plan["members"], 1):
                        acquired = acquire_member(archive, member, destinations[member["path"]])
                        report["acquired"][member["path"]] = acquired
                        if n % 3 == 0:
                            print(json.dumps({"members_complete": n, "members_total": len(plan["members"]),
                                              "range_bytes": stream.transferred}), flush=True)
                        (output / "acquisition.json").write_text(json.dumps(report, indent=2))
            finally:
                report["archive_range_bytes"] = stream.transferred
        stage = "preprocessing"
        manifest = assemble_manifest(plan, review, report["acquired"])
        manifest_path = output / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        processed = root / f"data/processed/e0-{os.environ['SLURM_JOB_ID']}"
        artifact = prepare(manifest, processed)
        if [c["action_id"] for c in artifact["classes"]] != [c["action_id"] for c in plan["class_mapping"]]:
            raise RuntimeError("Prepared vocabulary differs from acquisition plan")
        report.update(dataset=str(processed / "dataset.json"), summary=artifact["summary"], training_ready=True)
    except Exception as error:
        # Never log remote exception text containing signed URLs or credentials.
        report["failure"] = {"type": type(error).__name__, "stage": stage,
                             "reason": str(error) if stage == "preprocessing" and isinstance(error, ValueError)
                             else "Stopped before training; completed CRC-verified files are reusable"}
    finally:
        (output / "acquisition.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"report": str(output / "acquisition.json"), **{k: v for k, v in report.items() if k != "acquired"}}, indent=2), flush=True)
    if "failure" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
