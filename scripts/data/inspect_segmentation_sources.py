"""CPU-Slurm metadata inspection; never downloads videos or model weights."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import urllib.request


DATASET_REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
TOOLKIT_REVISION = "8d4b7f1bce3f99874514199aae7504f78ebf3517"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
CAMERAS = ["C10095", "C10115", "C10118", "C10119", "C10379", "C10390", "C10395", "C10404"]


def small_json(url, limit=300_000):
    request = urllib.request.Request(url, headers={"User-Agent": "JointPoseAction-research-pilot"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise RuntimeError("Small metadata response exceeded its explicit bound")
    return json.loads(raw), raw


def main():
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction") or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dedicated project CPU Slurm execution required")
    if not Path(os.environ.get("HF_HOME", "/nonexistent")).resolve().is_relative_to(root):
        raise RuntimeError("Dedicated Hugging Face cache required")
    output = root / "experiments/segmentation_pilot" / ("discovery-" + os.environ["SLURM_JOB_ID"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "job_id": os.environ["SLURM_JOB_ID"],
              "dataset_revision": DATASET_REVISION, "toolkit_revision": TOOLKIT_REVISION,
              "model_revision": SAM3_REVISION, "large_payload_downloaded": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cameras": CAMERAS, "candidates": [], "calibration_files": []}
    try:
        stat = os.statvfs(root)
        report["storage"] = {"filesystem_capacity_bytes": stat.f_blocks * stat.f_frsize,
                             "filesystem_available_bytes": stat.f_bavail * stat.f_frsize,
                             "dedicated_allocation_limit_bytes": 50_000_000_000}
        for key in ("ceph.quota.max_bytes", "ceph.dir.rbytes"):
            try:
                report["storage"][key] = int(os.getxattr(root, key).decode())
            except OSError:
                report["storage"][key] = None
        report["packages"] = {}
        for name in ("torch", "torchvision", "numpy", "opencv-python", "huggingface_hub", "av"):
            try:
                report["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["packages"][name] = None
        dataset_path = root / "data/processed/e0-242798/dataset.json"
        raw = dataset_path.read_bytes()
        expected = "f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59"
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError("Existing frozen recognition manifest changed")
        frozen = json.loads(raw)
        splits = {s["recording_id"]: s["split"] for s in frozen["segments"]}
        report["recognition_manifest_sha256"] = expected
        from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url
        token = get_token()
        report["private_hf_token_present"] = bool(token)
        try:
            meta = get_hf_file_metadata(hf_hub_url("facebook/sam3", "sam3.pt", revision=SAM3_REVISION), token=token)
            report["sam3_checkpoint_access"] = {"accessible": True, "bytes": meta.size, "etag": meta.etag}
        except Exception as error:
            report["sam3_checkpoint_access"] = {"accessible": False, "error_type": type(error).__name__,
                "http_status": getattr(getattr(error, "response", None), "status_code", None)}
        listing, _ = small_json("https://api.github.com/repos/facebookresearch/assemblyhands-toolkit/contents/calib/nimble_json_calib?ref=" + TOOLKIT_REVISION)
        candidates = []
        for item in listing:
            if item["name"].endswith(".json") and item["size"] < 65536:
                rec = item["name"][:-5]
                candidates.append((splits.get(rec) != "train", rec, item))
        api = HfApi(token=token)
        for _, rec, item in sorted(candidates):
            url = "https://raw.githubusercontent.com/facebookresearch/assemblyhands-toolkit/" + TOOLKIT_REVISION + "/calib/nimble_json_calib/" + item["name"]
            _, raw = small_json(url, limit=65536)
            target = output / "calibration" / item["name"]
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(raw)
            report["calibration_files"].append({"recording_id": rec, "path": str(target), "bytes": len(raw),
                "source_url": url, "sha256": hashlib.sha256(raw).hexdigest(), "recognition_split": splits.get(rec)})
            if splits.get(rec) != "train" or len(report["candidates"]) >= 5:
                continue
            files = list(api.list_repo_tree("cvml-nus/assembly101", repo_type="dataset", revision=DATASET_REVISION,
                                           path_in_repo="recordings/" + rec))
            views = [{"camera_id": Path(f.path).name.split("_")[0], "path": f.path, "bytes": f.size}
                     for f in files if Path(f.path).name in {c + "_rgb.mp4" for c in CAMERAS}]
            report["candidates"].append({"recording_id": rec, "recognition_split": splits[rec],
                "calibration_path": str(target), "views": views, "all_eight_views_available": sorted(x["camera_id"] for x in views) == CAMERAS,
                "full_video_bytes": sum(x["bytes"] for x in views), "status": "unvalidated_candidate_not_selected"})
        report["small_calibration_bytes_downloaded"] = sum(x["bytes"] for x in report["calibration_files"])
        report["remaining_gates"] = ["real multiview visual target selection", "recording-specific projection/distortion checks",
            "independent temporal correspondence evidence", "storage-approved bounded frame acquisition", "SAM3 checkpoint access"]
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "reason": "Discovery stopped; no video/checkpoint acquisition attempted"}
    finally:
        (output / "discovery.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"report": str(output / "discovery.json"), **report}, indent=2), flush=True)
    if "failure" in report:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
