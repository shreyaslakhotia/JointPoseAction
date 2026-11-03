"""Bounded cache/adapter for the frozen E0 rows and official HandFormer sampler.

No new examples or labels are selected. The cache contains source observations,
not resized segment tensors, so training crops precede temporal interpolation.
"""
import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from e0.data import pose_bounds, read_cleaned, sha256
from e0.quality import load_confidences


CONTRACT = "handformer-e0-source-observations-v1"
MAX_CACHE_BYTES = 6_000_000_000


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _usage(root):
    # pathlib follows the dedicated root symlink after resolve; no other project.
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def prepare(dataset_path, output, max_cache_bytes=MAX_CACHE_BYTES, project_root=None,
            allocation_bytes=50_000_000_000, headroom_bytes=2_000_000_000):
    dataset_path, output = Path(dataset_path).resolve(), Path(output).resolve()
    artifact = json.loads(dataset_path.read_text(encoding="utf-8"))
    if artifact.get("ready_for_training") is not True:
        raise ValueError("Frozen source dataset is not ready")
    if not 0 < max_cache_bytes <= MAX_CACHE_BYTES:
        raise ValueError("Cache bound must not exceed 6 GB")
    root = Path(project_root).resolve() if project_root else None
    if root and (not output.is_relative_to(root) or not dataset_path.is_relative_to(root)):
        raise ValueError("Dataset and cache must stay inside dedicated project root")
    digest = sha256(dataset_path)
    manifest_path = output / "compatibility.json"
    progress_path = output / "cache-progress.json"
    if manifest_path.exists():
        existing = load_compatibility(manifest_path, dataset_path, artifact, verify_hashes=True)
        return existing
    output.mkdir(parents=True, exist_ok=True)
    existing = None
    if progress_path.exists():
        existing = load_compatibility(progress_path, dataset_path, artifact,
                                      verify_hashes=True, require_ready=False)
    elif any(output.iterdir()):
        raise ValueError("Partial cache exists without completion manifest; inspect before retry")
    used = _usage(root) if root else 0
    cached = existing["cache_bytes"] if existing else 0
    available = min(max_cache_bytes, shutil.disk_usage(output).free - headroom_bytes + cached,
                    allocation_bytes - used - headroom_bytes + cached if root else max_cache_bytes)
    if available <= 0:
        raise ValueError("Insufficient dedicated project storage headroom")
    grouped = {}
    for row in artifact["segments"]:
        grouped.setdefault(row["recording_id"], []).append(row)
    result = {"contract": CONTRACT, "ready": False, "dataset_path": str(dataset_path),
              "dataset_sha256": digest, "classes": artifact["classes"],
              "segment_ids": [r["segment_id"] for r in artifact["segments"]],
              "split_counts": dict(Counter(r["split"] for r in artifact["segments"])),
              "additional_exclusions": [], "recordings": {}, "cache_bytes": 0,
              "segment_missing_observations": {},
              "cache_bound_bytes": max_cache_bytes, "available_bytes_at_start": available,
              "normalization": "original_E0_segment_origin_and_scale_world_axes",
              "confidence_policy": "positive_source_hand_score_and_finite_joint",
              "absolute_video_clock_verified": False, "physical_units_verified": False}
    if existing:
        result = existing
    result.setdefault("segment_missing_observations", {})
    _json(progress_path, result)
    for position, (recording, rows) in enumerate(sorted(grouped.items())):
        if recording in result["recordings"]:
            continue
        entry = artifact["source_manifest"]["recordings"][recording]
        for source_path in (entry["path"], entry["confidence"]["path"]):
            if root and not Path(source_path).resolve().is_relative_to(root):
                raise ValueError("Raw source is outside dedicated project")
        if sha256(entry["path"]) != entry["sha256"] or sha256(entry["confidence"]["path"]) != entry["confidence"]["sha256"]:
            raise ValueError("Raw source hash mismatch")
        frames = read_cleaned(entry["path"])
        scores = load_confidences(entry["confidence"]["path"])
        ids = np.asarray(sorted(frames), dtype=np.int64)
        if set(scores) != set(frames):
            raise ValueError("Confidence/source frame keys differ")
        # Exact future cache sizes, including three small .npy headers (<1 KB).
        predicted = len(ids) * (2 * 21 * 3 * 4 + 2 * 21 + 8) + 1024
        if result["cache_bytes"] + predicted > available:
            raise ValueError("Compact source cache would exceed storage bound")
        xyz = np.stack([frames[int(i)] for i in ids]).astype(np.float32)
        confidence = np.stack([scores[int(i)] for i in ids])
        valid = np.isfinite(xyz).all(-1) & (confidence[..., None] > 0)
        xyz = np.where(valid[..., None], xyz, 0).astype(np.float32)
        for row in rows:
            first, stop = pose_bounds(row, artifact["source_manifest"]["pose_profile"])
            if first < ids[0] or stop - 1 > ids[-1]:
                raise ValueError("A frozen E0 example lacks full interval source observations")
            q = row["quality"]
            interval = np.arange(first, stop)
            missing = interval[~np.isin(interval, ids)]
            if len(missing):
                result["segment_missing_observations"][row["segment_id"]] = {
                    "nominal_frames": stop - first, "missing_source_frames": len(missing),
                    "missing_original_E0_sample_positions": int(np.isin(q["pose_frame_indices"], missing).sum())}
            if not np.isfinite(q["source_to_feature_scale"]) or q["source_to_feature_scale"] <= 0 or np.asarray(q["origin_feature_units"]).shape != (3,):
                raise ValueError("Invalid original E0 segment normalization")
        prefix = f"recording-{position:04d}"
        files = {}
        for key, array in (("xyz", xyz), ("valid", valid), ("frame_ids", ids)):
            path = output / f"{prefix}-{key}.npy"
            np.save(path, array, allow_pickle=False)
            files[key] = {"path": path.name, "sha256": sha256(path), "shape": list(array.shape),
                          "dtype": str(array.dtype), "bytes": path.stat().st_size}
        result["recordings"][recording] = {"files": files, "source_sha256": entry["sha256"],
            "confidence_sha256": entry["confidence"]["sha256"], "source_frame_count": len(ids),
            "missing_nominal_source_frames": int(ids[-1] - ids[0] + 1 - len(ids))}
        result["cache_bytes"] += sum(f["bytes"] for f in files.values())
        _json(progress_path, result)
        print(json.dumps({"handformer_cached_recordings": position + 1,
                          "total_recordings": len(grouped), "cache_bytes": result["cache_bytes"]}), flush=True)
        del xyz, frames, scores, confidence, valid
    result["ready"] = True
    result["missing_observation_totals"] = {
        "recordings_with_missing_nominal_frames": sum(x["missing_nominal_source_frames"] > 0 for x in result["recordings"].values()),
        "segments_with_missing_source_frames": len(result["segment_missing_observations"]),
        "missing_source_frame_positions_across_segments": sum(x["missing_source_frames"] for x in result["segment_missing_observations"].values()),
        "missing_original_E0_sample_positions": sum(x["missing_original_E0_sample_positions"] for x in result["segment_missing_observations"].values())}
    _json(manifest_path, result)
    return result


def load_compatibility(path, dataset_path, artifact, verify_hashes=False, require_ready=True):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract") != CONTRACT or (require_ready and value.get("ready") is not True):
        raise ValueError("HandFormer cache has not completed")
    if value["dataset_sha256"] != sha256(dataset_path) or value["classes"] != artifact["classes"]:
        raise ValueError("HandFormer dataset/class identity differs from frozen E0")
    if value["segment_ids"] != [x["segment_id"] for x in artifact["segments"]] or value["additional_exclusions"]:
        raise ValueError("HandFormer may not silently change the evaluation population")
    if value["cache_bytes"] > MAX_CACHE_BYTES:
        raise ValueError("Cache exceeds recorded storage bound")
    for item in value["recordings"].values():
        for file in item["files"].values():
            target = (path.parent / file["path"]).resolve()
            if not target.is_relative_to(path.parent) or not target.is_file():
                raise ValueError("Unsafe/missing compact cache path")
            if verify_hashes and sha256(target) != file["sha256"]:
                raise ValueError("Compact cache hash mismatch")
    return value


def crop_and_resize(data, training, samples=120, interval=(0.70, 1.0), evaluation_fraction=0.85):
    """Author temporal policy, with explicit length instead of coordinate sums.

    data is C,T,V,M. Unlike the released helper, a valid zero-coordinate frame
    remains in the temporal axis; confidence was explicitly handled upstream.
    """
    c, t, v, m = data.shape
    if (c, v, m) != (3, 21, 2) or t < 1:
        raise ValueError("Expected a nonempty C3,T,V21,M2 pose segment")
    if training:
        fraction = float(np.random.uniform(*interval))
        length = min(max(int(np.floor(t * fraction)), 60), t)
        offset = int(np.random.randint(0, t - length + 1))
        selected = data[:, offset:offset + length]
    else:
        bias = int((1 - evaluation_fraction) * t / 2)
        selected = data[:, bias:t - bias]
    valid_size = selected.shape[1]
    tensor = torch.as_tensor(selected.copy(), dtype=torch.float32)
    tensor = tensor.permute(0, 2, 3, 1).contiguous().view(c * v * m, valid_size)
    tensor = F.interpolate(tensor[None, None], size=(c * v * m, samples),
                           mode="bilinear", align_corners=False)
    return tensor.reshape(c, v, m, samples).permute(0, 3, 1, 2).contiguous()


class HandFormerDataset(Dataset):
    def __init__(self, dataset_path, artifact, split, config, compatibility):
        self.rows = [r for r in artifact["segments"] if r["split"] == split]
        self.profile = artifact["source_manifest"]["pose_profile"]
        self.path = Path(compatibility).resolve()
        self.cache = load_compatibility(self.path, dataset_path, artifact)
        self.training = split == "train" and not config.get("sanity_mode", False)
        self.preprocessing = config.get("preprocessing", {})
        self._opened = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def close(self):
        for arrays in self._opened.values():
            for array in arrays.values():
                array._mmap.close()
        self._opened.clear()

    def __getitem__(self, index):
        row = self.rows[index]
        recording = row["recording_id"]
        if recording not in self._opened:
            files = self.cache["recordings"][recording]["files"]
            self._opened[recording] = {k: np.load(self.path.parent / f["path"], mmap_mode="r", allow_pickle=False)
                                       for k, f in files.items()}
            if len(self._opened) > 8:
                self._opened.popitem(last=False)
        arrays = self._opened[recording]
        first, stop = pose_bounds(row, self.profile)
        start, end = np.searchsorted(arrays["frame_ids"], [first, stop])
        if start >= len(arrays["frame_ids"]) or first < arrays["frame_ids"][0] or stop - 1 > arrays["frame_ids"][-1]:
            raise ValueError("Cached frame interval differs from frozen source example")
        q = row["quality"]
        xyz = np.zeros((stop - first, 2, 21, 3), dtype=np.float64)
        valid = np.zeros((stop - first, 2, 21), dtype=bool)
        positions = arrays["frame_ids"][start:end] - first
        xyz[positions] = arrays["xyz"][start:end]
        valid[positions] = arrays["valid"][start:end]
        xyz *= q["source_to_feature_scale"]
        xyz -= np.asarray(q["origin_feature_units"])[None, None, None]
        xyz = np.where(valid[..., None], xyz, 0).astype(np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError("Nonfinite HandFormer adapter input")
        data = xyz.transpose(3, 0, 2, 1)
        tensor = crop_and_resize(data, self.training,
            samples=self.preprocessing.get("samples", 120),
            interval=self.preprocessing.get("train_crop_interval", [0.70, 1.0]),
            evaluation_fraction=1.0 if self.preprocessing.get("sanity_full_segment") else self.preprocessing.get("evaluation_crop_fraction", 0.85))
        return tensor, row["label"], index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--max-cache-bytes", type=int, default=MAX_CACHE_BYTES)
    args = parser.parse_args()
    result = prepare(args.dataset, args.output, args.max_cache_bytes, args.project_root)
    print(json.dumps({"compatibility": str(Path(args.output).resolve() / "compatibility.json"),
                      "cache_bytes": result["cache_bytes"], "split_counts": result["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
