"""Pin and list narrow Assembly101 metadata paths; never download data blobs."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    # Authentication/cache files must stay in the dedicated project.
    expected = Path("/projects/_ssd/jointposeaction")
    root = Path("/projects/jointposeaction").resolve()
    cache = Path(os.environ.get("HF_HOME", "/nonexistent")).resolve()
    if root != expected or not cache.is_relative_to(root):
        raise RuntimeError("Source scripts/cluster/env.sh before running on project storage")
    from huggingface_hub import HfApi

    api = HfApi()
    repo = "cvml-nus/assembly101"
    revision = api.repo_info(repo_id=repo, repo_type="dataset", revision=args.revision).sha
    roots = list(api.list_repo_tree(repo, repo_type="dataset", revision=revision))
    rows = []
    if not any(item.path == "annotations" for item in roots):
        raise RuntimeError("Expected annotations folder absent; inspect upstream layout")
    for item in api.list_repo_tree(repo, path_in_repo="annotations", recursive=True,
                                   repo_type="dataset", revision=revision):
        if hasattr(item, "size"):
            rows.append({"path": item.path, "size_bytes": item.size,
                         "git_blob_id": getattr(item, "blob_id", None)})
        if len(rows) > 10000:
            raise RuntimeError("Unexpected metadata inventory size; stop and review")
    print(json.dumps({"repo": repo, "revision": revision,
                      "root_paths": [item.path for item in roots],
                      "annotation_files": rows,
                      "downloaded_data_files": 0}, indent=2))


if __name__ == "__main__":
    main()
