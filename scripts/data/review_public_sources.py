"""Bounded public source review. Never fetches a pose archive, video or split CSV."""
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

OUT = Path("data/e0_source_review")


def get(url, limit=2_000_000):
    with urlopen(Request(url, headers={"User-Agent": "JointPoseAction-source-review"}), timeout=30) as response:
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise RuntimeError(f"Bounded response exceeded: {url}")
        return raw


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {"date": "2026-11-02", "github": {}, "hf": {}, "downloaded_pose_or_video_bytes": 0}
    for repo in ("s-shamil/HandFormer", "assembly-101/assembly101-action-recognition"):
        info = json.loads(get(f"https://api.github.com/repos/{repo}/commits/main"))
        sha = info["sha"]
        tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"))
        paths = [x["path"] for x in tree["tree"] if x["path"].endswith((".py", ".md", ".txt"))]
        evidence["github"][repo] = {"revision": sha, "source_paths": paths}
        for path in paths:
            if (repo.endswith("HandFormer") and path in ("Preprocess/1_get_clip_data.py", "Preprocess/config.py", "Preprocess/utils.py")) or ("MS-G3D" in path and any(x in path.lower() for x in ("gen", "readme", "assembly", "graph"))):
                raw = get(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}", 500_000)
                target = OUT / repo.split("/")[-1] / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    repo = "cvml-nus/assembly101"
    info = json.loads(get(f"https://huggingface.co/api/datasets/{repo}"))
    sha = info["sha"]
    root = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/tree/{sha}"))
    fine = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/tree/{sha}/annotations/fine-grained-annotations"))
    evidence["hf"] = {"revision": sha, "gated": info.get("gated"),
                      "fine_grained_files": fine,
                      "pose_archive": [x for x in root if x["path"] == "AssemblyPoses.zip"]}
    path = "annotations/README.md"
    try:
        raw = get(f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{path}", 20_000)
        (OUT / "annotations_README.md").write_bytes(raw)
        evidence["hf"]["small_file_probe"] = {"path": path, "http_status": 200,
                                                 "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    except HTTPError as error:
        evidence["hf"]["small_file_probe"] = {"path": path, "http_status": error.code}
    target = Path("reproducibility/datasets/e0_source_evidence.json")
    target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
