"""Inspect size-reported timestamp/confidence/wrist-transform companions, CPU only."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import zipfile

import numpy as np

from e0.acquisition import RangeFile, validate_range_headers
from e0.data import read_cleaned, sha256
from scripts.data.inspect_e0_sources import REVISION


KINDS = {"timestamp", "hand_confidences", "xf_transf"}
EXPECTED_BYTES = (1335648, 3621046)


def summarize(value, frames, kind):
    """Describe observed structures; no threshold, time scale or validity is inferred."""
    result = {"top_level_type": type(value).__name__, "tracking_mask_applied": False}
    if not isinstance(value, dict) or not value:
        result["bounded_preview"] = repr(value)[:2000]
        return result
    result["entries"] = len(value)
    numeric = all(isinstance(k, str) and k.isdigit() for k in value)
    keys = sorted(value, key=int) if numeric else sorted(value)
    result["keys_are_numeric_frame_ids"] = numeric
    examples = {k: value[k] for k in dict.fromkeys(keys[:2] + keys[-2:])}
    preview = json.dumps(examples, allow_nan=False)
    result["example_json_prefix"] = preview[:4000]
    result["example_preview_truncated"] = len(preview) > 4000
    if not numeric:
        return result
    observed = {int(k): v for k, v in value.items()}
    shared = sorted(set(observed) & set(frames))
    result.update(minimum_frame_id=int(keys[0]), maximum_frame_id=int(keys[-1]),
                  pose_keys_with_companion=len(shared), pose_key_count=len(frames),
                  companion_keys_without_pose=len(set(observed) - set(frames)))
    if kind == "timestamp" and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()):
        times = [value[k] for k in keys]
        # Subtract Python integers before float conversion to retain large-clock increments.
        delta = np.asarray([b - a for a, b in zip(times, times[1:])], dtype=float)
        result["scalar_timestamp_diagnostics"] = {
            "first_raw": times[0], "last_raw": times[-1], "span_source_units": times[-1] - times[0],
            "nonpositive_steps": int((delta <= 0).sum()),
            "step_quantiles_source_units": np.quantile(delta, [0, .5, 1]).tolist() if len(delta) else [],
            "timestamp_units_and_video_clock_origin_verified": False}
    if kind == "hand_confidences" and shared:
        rows = [observed[k] for k in shared]
        scalar_hands = all(isinstance(row, dict) and set(row) == {"0", "1"}
                           and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row.values())
                           for row in rows)
        if scalar_hands:
            associations = []
            for h in range(2):
                scores = np.array([observed[k][str(h)] for k in shared], dtype=float)
                xyz = np.stack([frames[k][h] for k in shared])
                zero_wrist = (xyz[:, 5] == 0).all(-1)
                finite = scores[np.isfinite(scores)]
                associations.append({"hand_key": str(h), "finite_scores": int(np.isfinite(scores).sum()),
                    "score_quantiles": np.quantile(finite, [0, .5, 1]).tolist() if len(finite) else [],
                    "scores_equal_zero": int((scores == 0).sum()),
                    "zero_wrist_and_zero_score": int((zero_wrist & (scores == 0)).sum()),
                    "zero_wrist_and_nonzero_score": int((zero_wrist & (scores != 0)).sum()),
                    "nonzero_wrist_and_zero_score": int((~zero_wrist & (scores == 0)).sum())})
            result["observed_score_zero_wrist_associations"] = associations
            result["score_semantics_and_threshold_verified"] = False
    if kind == "xf_transf" and shared:
        rows = [observed[k] for k in shared]
        if all(isinstance(row, dict) and set(row) == {"0", "1"} for row in rows):
            matrices = np.asarray([[row["0"], row["1"]] for row in rows], dtype=float)
            result["observed_transform_array_shape"] = list(matrices.shape)
            if matrices.shape == (len(shared), 2, 4, 4) and np.isfinite(matrices).all():
                wrists = np.stack([frames[k][:, 5] for k in shared])
                residual = np.linalg.norm(matrices[:, :, :3, 3] - wrists, axis=-1)
                result["translation_minus_pose_wrist_source_unit_quantiles"] = np.quantile(residual, [0, .5, 1]).tolist()
                result["identity_transform_counts_by_hand"] = np.isclose(matrices, np.eye(4), atol=1e-8, rtol=0).all((2, 3)).sum(0).tolist()
                result["transform_direction_and_physical_units_verified"] = False
    return result


def main():
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction") or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dedicated project CPU Slurm execution required")
    if not Path(os.environ.get("HF_HOME", "/nonexistent")).resolve().is_relative_to(root):
        raise RuntimeError("Dedicated HF cache required")
    source_path = root / "data/annotations/e0-samples-242734/report.json"
    source = json.loads(source_path.read_text())
    if source["revision"] != REVISION or "failure" in source or len(source["records"]) != 3:
        raise RuntimeError("Complete validated sample report required")
    records = {r["recording_id"]: r for r in source["records"]}
    frames = {}
    for rec, row in records.items():
        if not row["zip_crc_verified"] or sha256(row["path"]) != row["sha256"]:
            raise RuntimeError("Verified pose sample changed")
        frames[rec] = read_cleaned(row["path"])
    members = [m for m in source["companion_inventory"] if Path(m["path"]).parent.name in KINDS]
    if len(members) != 9 or (sum(m["compressed_bytes"] for m in members), sum(m["uncompressed_bytes"] for m in members)) != EXPECTED_BYTES:
        raise RuntimeError("Reported companion size plan changed")
    if {(Path(m["path"]).stem, Path(m["path"]).parent.name) for m in members} != {(r, k) for r in records for k in KINDS}:
        raise RuntimeError("Companion membership mismatch")
    output = root / f"data/annotations/e0-companions-{os.environ['SLURM_JOB_ID']}"
    output.mkdir(exist_ok=False)
    report = {"revision": REVISION, "source_report_sha256": sha256(source_path),
              "planned_compressed_bytes": EXPECTED_BYTES[0], "planned_extracted_bytes": EXPECTED_BYTES[1],
              "range_budget_bytes": 3000000, "records": [], "training_ready": False}
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
            stream = RangeFile(metadata.size, fetch, max_total=3000000)
            try:
                with zipfile.ZipFile(stream) as archive:
                    for m in members:
                        info = archive.getinfo(m["path"])
                        if (info.compress_size, info.file_size) != (m["compressed_bytes"], m["uncompressed_bytes"]):
                            raise RuntimeError("Member size changed")
                        kind, rec = Path(m["path"]).parent.name, Path(m["path"]).stem
                        path = output / f"{kind}--{rec}.json"
                        with archive.open(info) as reader, path.open("xb") as writer:
                            size = 0
                            while block := reader.read(65536):
                                size += len(block)
                                if size > info.file_size:
                                    raise RuntimeError("Expanded member limit exceeded")
                                writer.write(block)
                        if size != info.file_size:
                            raise RuntimeError("Truncated member")
                        result = {"recording_id": rec, "kind": kind, "path": str(path),
                                  "sha256": sha256(path), "zip_crc_verified": True,
                                  **summarize(json.loads(path.read_text()), frames[rec], kind)}
                        report["records"].append(result)
                        print(json.dumps(result), flush=True)
            finally:
                report["archive_range_bytes"] = stream.transferred
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "reason": "Companion inspection stopped; no training attempted"}
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"report": str(output / "report.json"), **{k: v for k, v in report.items() if k != "records"}}), flush=True)
    if "failure" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
