"""Cluster CPU inspection: approved CSVs and at most 4 MB of archive HTTP ranges."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from e0.acquisition import RangeFile, inspect_archive, validate_range_headers
from e0.data import check_leakage, load_actions, load_segments, read_cleaned, sha256

REVISION = "bfc15ea5e3f0bc8f8c232af6c1b45aa137a9d967"
EXPECTED = {"actions.csv": 81530, "head_actions.txt": 3139, "test.csv": 47993475,
            "train.csv": 100677017, "validation.csv": 34544546}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-approved-splits", action="store_true")
    args = parser.parse_args()
    root = Path("/projects/jointposeaction").resolve()
    if root != Path("/projects/_ssd/jointposeaction") or not Path(os.environ.get("HF_HOME", "/nonexistent")).resolve().is_relative_to(root):
        raise RuntimeError("Source project cache configuration first")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run metadata processing in a CPU Slurm job")
    from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url, hf_hub_download
    import httpx
    token = get_token()
    if not token:
        raise RuntimeError("Authenticate privately using hf auth login after sourcing env.sh")
    output = root / "data/annotations" / f"e0-review-{os.environ['SLURM_JOB_ID']}"
    output.mkdir(exist_ok=False)
    report = {"revision": REVISION, "approved_metadata_bytes": sum(EXPECTED.values()),
              "pose_archive_bytes": 71972006749, "downloaded_videos": 0, "full_archive_downloaded": False}
    try:
        api = HfApi(token=token)
        prefix = "annotations/fine-grained-annotations"
        tree = {Path(x.path).name: x for x in api.list_repo_tree("cvml-nus/assembly101", repo_type="dataset", revision=REVISION, path_in_repo=prefix) if hasattr(x, "size")}
        for name, size in EXPECTED.items():
            if name not in tree or tree[name].size != size:
                raise RuntimeError("Pinned metadata layout/size changed; stop before acquisition")
        names = list(EXPECTED) if args.download_approved_splits else ["actions.csv", "head_actions.txt"]
        report["files"] = []
        for name in names:
            path = hf_hub_download("cvml-nus/assembly101", filename=f"{prefix}/{name}",
                                   repo_type="dataset", revision=REVISION, token=token,
                                   local_dir=str(output / "official"))
            if Path(path).stat().st_size != EXPECTED[name]:
                raise RuntimeError("Downloaded metadata size mismatch")
            report["files"].append({"name": name, "path": path, "bytes": EXPECTED[name], "sha256": sha256(path)})
        if args.download_approved_splits:
            actions = load_actions(output / "official" / prefix / "actions.csv")
            segments, exclusions = load_segments({s: output / "official" / prefix / f"{s}.csv" for s in ("train", "validation", "test")}, actions)
            from collections import Counter
            report["coverage"] = {"mapped_classes": len(actions),
                                  "canonical_segments": dict(Counter(x["split"] for x in segments)),
                                  "recordings": {s: sorted({x["recording_id"] for x in segments if x["split"] == s}) for s in ("train", "validation", "test")},
                                  "participants": {s: sorted({x["participant_id"] for x in segments if x["split"] == s}) for s in ("train", "validation", "test")},
                                  "alternate_view_rows": len(exclusions),
                                  "leakage": check_leakage(segments, "report_official_overlap")}
        url = hf_hub_url("cvml-nus/assembly101", "AssemblyPoses.zip", repo_type="dataset", revision=REVISION)
        metadata = get_hf_file_metadata(url, token=token)
        if metadata.size != report["pose_archive_bytes"]:
            raise RuntimeError("Archive size differs from pinned inventory")
        headers = {"Authorization": f"Bearer {token}"} if urlparse(metadata.location).hostname == "huggingface.co" else {}
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            def fetch(start, count):
                request_headers = {**headers, "Range": f"bytes={start}-{start + count - 1}", "Accept-Encoding": "identity"}
                with client.stream("GET", metadata.location, headers=request_headers) as response:
                    validate_range_headers(response.status_code, response.headers.get("Content-Range"), start, count, metadata.size)
                    chunks, size = [], 0
                    for block in response.iter_bytes(chunk_size=65536):
                        size += len(block)
                        if size > count:
                            raise RuntimeError("Range body exceeded advertised bounds")
                        chunks.append(block)
                    return b"".join(chunks)
            stream = RangeFile(metadata.size, fetch)
            try:
                report["archive"] = inspect_archive(stream)
                sample_path = output / "cleaned_first_frames.json"
                sample_path.write_text(json.dumps(report["archive"].pop("sample_frames")))
                frames = read_cleaned(sample_path)
                report["archive"].update(sample_path=str(sample_path),
                                          sample_sha256=sha256(sample_path),
                                          sampled_frame_ids=sorted(frames), sample_shapes=[2, 21, 3])
            finally:
                report["archive_range_bytes"] = stream.transferred
    except Exception as error:
        # Never log opaque signed URLs, tokens, SDK response bodies or exception
        # strings that can contain authorization material.
        report["failure"] = {"type": type(error).__name__,
                             "reason": str(error) if isinstance(error, RuntimeError) else "Remote access/schema inspection failed; inspect locally without sharing credentials"}
        raise SystemExit(1)
    finally:
        (output / "inspection.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
