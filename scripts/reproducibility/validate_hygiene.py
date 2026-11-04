"""Read-only hygiene checks: preservation, research interfaces and navigation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def git(*args, **kwargs):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
                                   cwd=ROOT, **kwargs)


def validate():
    manifest = json.loads((ROOT / "reproducibility/manifests/repository_hygiene.json").read_text(encoding="utf-8"))
    failures, raw_matches, checkout_equivalents, protected = [], 0, 0, 0
    for row in manifest["files"]:
        path = (ROOT / row["final_path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(f"Missing/unsafe preservation destination: {row['original_path']}")
            continue
        raw = path.read_bytes()
        sha = lambda data: hashlib.sha256(data).hexdigest()
        if sha(raw) == row["original_working_sha256"]:
            raw_matches += 1
        elif sha(raw) == row["original_sha256"] or sha(raw.replace(b"\r\n", b"\n")) == row["original_sha256"]:
            checkout_equivalents += 1
        else:
            failures.append(f"Original content changed: {row['original_path']}")
        # Existing model, preprocessing, training, evaluation, configurations,
        # tests and provenance remain at exactly their prior entry points.
        original = row["original_path"]
        if original.startswith(("e0/", "phase1c/", "configs/", "tests/", "reproducibility/")):
            protected += 1
            if row["final_path"] != original:
                failures.append(f"Protected research path moved: {original}")

    # Also retain the later completed-result revision, rather than validating
    # only the earlier pre-completion hygiene baseline.
    completed = ROOT / "reproducibility/manifests/main_consolidation.json"
    completed_matches = 0
    if completed.is_file():
        for row in json.loads(completed.read_text(encoding="utf-8"))["files"]:
            path = (ROOT / row["final_path"]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                failures.append(f"Missing completed-result destination: {row['original_path']}")
                continue
            raw = path.read_bytes()
            exact = hashlib.sha256(raw).hexdigest() == row["original_sha256"]
            equivalent = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == row["lf_equivalent_sha256"]
            if exact or equivalent:
                completed_matches += 1
            else:
                failures.append(f"Completed-result content changed: {row['original_path']}")

    broken_links = []
    docs = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md")), ROOT / "reproducibility/VALIDATION.md"]
    for path in docs:
        for match in re.finditer(r"\[[^\]\n]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = match.group(1).strip().split(' "', 1)[0].strip("<>").split("#", 1)[0]
            if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("/") or "*" in target:
                continue
            if not (path.parent / target).exists():
                broken_links.append(f"{path.relative_to(ROOT).as_posix()}: {target}")
    failures.extend(f"Broken current documentation link: {link}" for link in broken_links)

    probes = {
        "legacy_files/historical_code/3D_reconstruction/single_view/sam3d_repo/LICENSE": True,
        "legacy_files/historical_code/3D_reconstruction/single_view/sam3d/output/example.png": True,
        "legacy_files/old_experiments/mistake_repo/annots/example.csv": False,
        "legacy_files/old_experiments/mistake_repo/annots/example.bin": True,
        "legacy_files/old_experiments/experiments/sam2_test/deprecated/mask_overlays/C10118_0005_overlay.jpg": False,
        "legacy_files/old_experiments/experiments/sam2_test/deprecated/mask_overlays/C10095_0005_overlay.jpg": True,
        "legacy_files/old_experiments/experiments/reconstruction_data/session/rgb/frame.jpg": True,
        "experiments/phase1c/run/history.jsonl": True,
        "environments/example/metadata.json": True,
        "configs/b1_handformer.json": False,
    }
    for path, expected in probes.items():
        result = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}",
                                 "check-ignore", "--no-index", "--quiet", path], cwd=ROOT)
        if result.returncode not in (0, 1) or (result.returncode == 0) != expected:
            failures.append(f"Ignore policy mismatch: {path}")
    return {"baseline_commit": manifest["baseline_commit"],
            "validated_commit": git("rev-parse", "HEAD").decode().strip(),
            "branch": git("branch", "--show-current").decode().strip(),
            "original_files": len(manifest["files"]), "raw_byte_matches": raw_matches,
            "checkout_equivalent_matches": checkout_equivalents,
            "protected_research_and_provenance_files": protected,
            "completed_development_files_preserved": completed_matches,
            "current_documentation_files_checked": len(docs), "ignore_probes": len(probes),
            "failures": failures,
            "limitations": ["No remote checkpoint, raw dataset or Slurm job was accessed by this local hygiene check.",
                            "Preservation and import checks do not reproduce scientific recognition results."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional lightweight JSON evidence file")
    args = parser.parse_args()
    report = validate()
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
