"""Plan remaining pose+confidence+timestamp acquisition; ZIP directory only."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import zipfile

from e0.acquisition import RangeFile, validate_range_headers
from e0.data import check_leakage, choose_classes, load_actions, load_segments, sha256
from e0.quality import EXCLUDED_RECORDINGS
from scripts.data.inspect_e0_sources import REVISION


def build_plan(catalog, segments, actions):
    indexed = {(Path(m["path"]).parent.name, Path(m["path"]).stem): m for m in catalog}
    available = {rec for kind, rec in indexed if kind == "landmarks3D"} - set(EXCLUDED_RECORDINGS)
    missing = {rec: [kind for kind in ("landmarks3D", "timestamp", "hand_confidences")
                     if (kind, rec) not in indexed] for rec in available}
    missing = {rec: kinds for rec, kinds in missing.items() if kinds}
    eligible = available - set(missing)
    rows = [s for s in segments if s["recording_id"] in eligible]
    vocabulary = choose_classes(rows, min_train_examples=50, max_classes=100)
    rows = [s for s in rows if s["action_id"] in vocabulary]
    recordings = sorted({s["recording_id"] for s in rows})
    members = [indexed[kind, rec] for rec in recordings for kind in ("landmarks3D", "timestamp", "hand_confidences")]
    return {"excluded_recordings": EXCLUDED_RECORDINGS, "missing_companions": missing,
            "class_mapping": [actions[a] for a in vocabulary],
            "coverage": {split: {"recordings": len({s["recording_id"] for s in rows if s["split"] == split}),
                                  "segments_before_quality_checks": sum(s["split"] == split for s in rows)}
                         for split in ("train", "validation", "test")},
            "members": members,
            "compressed_bytes": sum(m["compressed_bytes"] for m in members),
            "uncompressed_bytes": sum(m["uncompressed_bytes"] for m in members),
            "bytes_by_kind": {kind: {key: sum(m[key] for m in members if Path(m["path"]).parent.name == kind)
                                     for key in ("compressed_bytes", "uncompressed_bytes")}
                              for kind in ("landmarks3D", "timestamp", "hand_confidences")},
            "training_ready": False, "payload_downloaded": False,
            "remaining_gates": ["60 FPS cadence and segment-bound checks for each acquired recording",
                                "positive-confidence masking and final class support after exclusions",
                                "validated feature units and reference temporal contract"]}


def main():
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction") or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dedicated project CPU Slurm execution required")
    if not Path(os.environ.get("HF_HOME", "/nonexistent")).resolve().is_relative_to(root):
        raise RuntimeError("Dedicated HF cache required")
    review_path = root / "data/annotations/e0-review-242218/inspection.json"
    review = json.loads(review_path.read_text())
    if review["revision"] != REVISION:
        raise RuntimeError("Pinned source revision mismatch")
    files = {entry["name"]: entry for entry in review["files"]}
    for entry in files.values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Annotation hash changed")
    actions = load_actions(files["actions.csv"]["path"])
    segments, _ = load_segments({s: files[f"{s}.csv"]["path"] for s in ("train", "validation", "test")}, actions)
    check_leakage(segments, "report_official_overlap")
    output = root / f"data/annotations/e0-acquisition-plan-{os.environ['SLURM_JOB_ID']}.json"
    report = {"revision": REVISION, "review_sha256": sha256(review_path), "training_ready": False}
    try:
        from huggingface_hub import get_token, get_hf_file_metadata, hf_hub_url
        import httpx
        token = get_token()
        if not token:
            raise RuntimeError("Private HF authentication required")
        metadata = get_hf_file_metadata(hf_hub_url("cvml-nus/assembly101", "AssemblyPoses.zip", repo_type="dataset", revision=REVISION), token=token)
        if metadata.size != 71972006749:
            raise RuntimeError("Archive size changed")
        headers = {"Authorization": f"Bearer {token}"} if urlparse(metadata.location).hostname == "huggingface.co" else {}
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            def fetch(start, count):
                with client.stream("GET", metadata.location, headers={**headers, "Range": f"bytes={start}-{start + count - 1}", "Accept-Encoding": "identity"}) as response:
                    validate_range_headers(response.status_code, response.headers.get("Content-Range"), start, count, metadata.size)
                    data = bytearray()
                    for block in response.iter_bytes(chunk_size=65536):
                        if len(data) + len(block) > count:
                            raise RuntimeError("Range response exceeded bound")
                        data.extend(block)
                    return bytes(data)
            stream = RangeFile(metadata.size, fetch, max_total=2000000)
            try:
                with zipfile.ZipFile(stream) as archive:
                    catalog = [{"path": m.filename, "compressed_bytes": m.compress_size, "uncompressed_bytes": m.file_size}
                               for m in archive.infolist() if not m.is_dir()]
                report.update(build_plan(catalog, segments, actions))
            finally:
                report["archive_range_bytes"] = stream.transferred
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "reason": "Acquisition planning stopped; no payload requested"}
    finally:
        with output.open("x", encoding="utf-8") as writer:
            json.dump(report, writer, indent=2)
        print(json.dumps({"report": str(output), **{k: v for k, v in report.items() if k not in ("members", "class_mapping")}}, indent=2), flush=True)
    if "failure" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
