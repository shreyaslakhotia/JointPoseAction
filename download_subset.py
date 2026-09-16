"""Download the initial Assembly101 video subset from Hugging Face.

Install the dependency with:
    pip install huggingface_hub

The repository is searched by filename so this script does not depend on a
particular directory layout in the Hugging Face dataset mirror.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

VIDEO_IDS = (
    "nusar-2021_action_both_9012-a16_9012_user_id_2021-02-01_162904",
    "nusar-2021_action_both_9024-b08b_9024_user_id_2021-02-23_150955",
    "nusar-2021_action_both_9053-c09a_9053_user_id_2021-02-08_134120",
)


def matching_files(repo_files: Iterable[str], video_id: str) -> list[str]:
    """Return files whose final path component is the requested video ID."""
    matches = []
    for repo_file in repo_files:
        filename = Path(repo_file).name
        if filename == video_id or filename.startswith(f"{video_id}."):
            matches.append(repo_file)
    return matches


def download_subset(repo_id: str, output_dir: Path, revision: str) -> None:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from error

    api = HfApi()
    print(f"Listing files in Hugging Face dataset: {repo_id}")
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for video_id in VIDEO_IDS:
        files = matching_files(repo_files, video_id)
        if not files:
            print(f"WARNING: no file found for {video_id}")
            continue

        for repo_file in files:
            print(f"Downloading {repo_file}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=repo_file,
                repo_type="dataset",
                revision=revision,
                local_dir=str(output_dir),
            )
            print(f"Finished: {local_path}")
            downloaded += 1

    print(f"Download complete. Downloaded {downloaded} file(s) to {output_dir}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="cvml-nus/assembly101",
        help="Hugging Face dataset repository ID (default: cvml-nus/assembly101)",
    )
    parser.add_argument("--revision", default="main", help="Dataset revision or branch")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw_videos"),
        help="Directory for downloaded files (default: data/raw_videos)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        download_subset(args.repo_id, args.output_dir, args.revision)
    except Exception as error:
        raise SystemExit(f"Download failed: {error}") from error