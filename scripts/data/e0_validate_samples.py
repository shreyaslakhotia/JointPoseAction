"""Acquire three size-pinned training pose members, then diagnose without training."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import zipfile

import numpy as np

from e0.acquisition import RangeFile, validate_range_headers
from e0.data import load_actions, load_segments, read_cleaned, sha256
from scripts.data.inspect_e0_sources import REVISION


EXPECTED_SIZES = [(642036, 3539785), (1665011, 4571939), (2631234, 7645468)]


def diagnose(frames, segments):
    """Numeric evidence only: zeros/stationarity are never tracking-validity labels."""
    keys = np.array(sorted(frames), dtype=np.int64)
    xyz = np.stack([frames[int(k)] for k in keys])
    finite = np.isfinite(xyz).all(-1)
    adjacent = np.diff(keys) == 1
    hands = []
    for h in range(2):
        unchanged = adjacent & np.isfinite(xyz[1:, h]).all((1, 2)) & np.isfinite(xyz[:-1, h]).all((1, 2))
        unchanged &= (xyz[1:, h] == xyz[:-1, h]).all((1, 2))
        longest = run = 0
        for same in unchanged:
            run = run + 1 if same else 0
            longest = max(longest, run)
        distance = np.linalg.norm(xyz[:, h, 8] - xyz[:, h, 5], axis=-1)
        distance = distance[np.isfinite(distance)]
        hands.append({"hand_key": str(h), "finite_joint_fraction": float(finite[:, h].mean()),
                      "all_zero_hand_frames": int((xyz[:, h] == 0).all((1, 2)).sum()),
                      "zero_wrist_frames": int((xyz[:, h, 5] == 0).all(-1).sum()),
                      "exactly_unchanged_adjacent_pairs": int(unchanged.sum()),
                      "longest_exactly_unchanged_run_frames": longest + 1 if longest else 0,
                      "wrist_to_index_proximal_source_unit_quantiles":
                          np.quantile(distance, [0, .5, 1]).tolist() if len(distance) else None})
    key_set = set(frames)
    mappings = []
    for base in (0, 1):
        for inclusive in (False, True):
            outside = missing = total = 0
            examples = []
            for s in segments:
                start = 2 * (s["start_frame"] - base)
                stop = 2 * (s["end_frame"] - base + int(inclusive))
                if start < keys[0] or stop - 1 > keys[-1] or stop <= start:
                    outside += 1
                n = max(0, stop - start)
                total += n
                missing += sum(k not in key_set for k in range(start, stop))
                if len(examples) < 4:
                    examples.append({"annotation_interval": [s["start_frame"], s["end_frame"]],
                                     "pose_half_open": [start, stop]})
            mappings.append({"annotation_index_base": base, "end_inclusive": inclusive,
                             "pose_frame_offset": 0, "segments_outside_bounds": outside,
                             "missing_pose_frames_with_overlap_counted": missing,
                             "requested_pose_frames_with_overlap_counted": total,
                             "examples": examples})
    return {"frames": len(keys), "minimum_frame_id": int(keys[0]), "maximum_frame_id": int(keys[-1]),
            "first_sorted_frame_ids": keys[:8].tolist(), "last_sorted_frame_ids": keys[-8:].tolist(),
            "missing_ids_inside_recording_bounds": int(keys[-1] - keys[0] + 1 - len(keys)),
            "hands": hands, "annotated_segments": len(segments), "candidate_mappings": mappings,
            "mapping_counts_do_not_establish_clock_or_endpoint_semantics": True,
            "units_verified": False, "tracking_validity_verified": False}


def main():
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction") or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dedicated project CPU Slurm execution required")
    if not Path(os.environ.get("HF_HOME", "/nonexistent")).resolve().is_relative_to(root):
        raise RuntimeError("Dedicated HF cache required")
    review_path = root / "data/annotations/e0-review-242218/inspection.json"
    feasibility_path = root / "data/annotations/e0-feasibility-242713.json"
    review, feasibility = (json.loads(p.read_text()) for p in (review_path, feasibility_path))
    if feasibility["review_sha256"] != sha256(review_path) or review["revision"] != REVISION:
        raise RuntimeError("Source provenance mismatch")
    train = set(feasibility["coverage"]["train"]["candidate_recordings"])
    members = sorted((m for m in feasibility["candidate_pose_members"] if m["recording_id"] in train),
                     key=lambda m: (m["compressed_bytes"], m["path"]))[:3]
    if [(m["compressed_bytes"], m["uncompressed_bytes"]) for m in members] != EXPECTED_SIZES:
        raise RuntimeError("Reported three-member size plan changed; no acquisition")
    files = {f["name"]: f for f in review["files"]}
    for name in ("train.csv", "actions.csv"):
        if sha256(files[name]["path"]) != files[name]["sha256"]:
            raise RuntimeError("Annotation hash changed")
    actions = load_actions(files["actions.csv"]["path"])
    segments, _ = load_segments({"train": files["train.csv"]["path"]}, actions)
    output = root / f"data/annotations/e0-samples-{os.environ['SLURM_JOB_ID']}"
    output.mkdir(exist_ok=False)
    report = {"revision": REVISION, "feasibility_sha256": sha256(feasibility_path),
              "planned_compressed_bytes": sum(x[0] for x in EXPECTED_SIZES),
              "planned_extracted_bytes": sum(x[1] for x in EXPECTED_SIZES),
              "range_budget_bytes": 8000000, "records": [], "training_ready": False}
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
        headers = {"Authorization": f"Bearer {token}"} if urlparse(metadata.location).hostname == "huggingface.co" else {}
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            def fetch(start, count):
                request = {**headers, "Range": f"bytes={start}-{start + count - 1}", "Accept-Encoding": "identity"}
                with client.stream("GET", metadata.location, headers=request) as response:
                    validate_range_headers(response.status_code, response.headers.get("Content-Range"), start, count, metadata.size)
                    data = bytearray()
                    for block in response.iter_bytes(chunk_size=65536):
                        if len(data) + len(block) > count:
                            raise RuntimeError("Range response exceeded limit")
                        data.extend(block)
                    return bytes(data)
            stream = RangeFile(metadata.size, fetch, max_total=8000000)
            try:
                with zipfile.ZipFile(stream) as archive:
                    # Companion files are inventoried only; sizes must precede acquisition.
                    report["companion_inventory"] = [
                        {"path": i.filename, "compressed_bytes": i.compress_size, "uncompressed_bytes": i.file_size}
                        for i in archive.infolist()
                        if Path(i.filename).stem in {m["recording_id"] for m in members}
                        and "/landmarks3D/" not in i.filename and not i.is_dir()]
                    for m in members:
                        info = archive.getinfo(m["path"])
                        if (info.compress_size, info.file_size) != (m["compressed_bytes"], m["uncompressed_bytes"]):
                            raise RuntimeError("Member size changed")
                        path = output / (m["recording_id"] + ".json")
                        # Fixed output basename, no ZIP extract/path traversal. EOF verifies CRC.
                        with archive.open(info) as source, path.open("xb") as target:
                            size = 0
                            while block := source.read(65536):
                                size += len(block)
                                if size > info.file_size:
                                    raise RuntimeError("Member expanded beyond bound")
                                target.write(block)
                        if size != info.file_size:
                            raise RuntimeError("Truncated member")
                        frames = read_cleaned(path)
                        rows = [s for s in segments if s["recording_id"] == m["recording_id"]]
                        result = {"recording_id": m["recording_id"], "path": str(path),
                                  "sha256": sha256(path), "zip_crc_verified": True,
                                  **diagnose(frames, rows)}
                        report["records"].append(result)
                        print(json.dumps(result), flush=True)
            finally:
                report["archive_range_bytes"] = stream.transferred
    except Exception as error:
        # HTTP exceptions may contain signed URLs; do not print exception text/traceback.
        report["failure"] = {"type": type(error).__name__, "reason": "Acquisition/validation stopped; no training attempted"}
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"report": str(output / "report.json"), **{k: v for k, v in report.items() if k not in ("records",)}}), flush=True)
    if "failure" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
