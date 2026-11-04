"""Read-only, dependency-free preservation checks. Run locally, not on login nodes."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT
    )


def main():
    tracked = git("ls-files", "-z").decode().strip("\0").split("\0")
    result = {"commit": git("rev-parse", "HEAD").decode().strip(),
              "preservation": {}, "small_artifacts": {}, "syntax": {},
              "patch_reconstruction": "Deferred: requires pinned upstream blobs"}
    failures = []
    for label, manifest, key in (
        ("preservation", "preservation_checksums.json", "preserved_as"),
        ("small_artifacts", "small_artifacts.json", "path"),
    ):
        counts = {"exact": 0, "crlf_only": 0, "deferred_patch_entries": 0}
        for entry in json.loads((ROOT / "reproducibility" / manifest).read_text())["files"]:
            if key not in entry:
                counts["deferred_patch_entries"] += 1
                continue
            path = entry[key]
            source = ROOT / path
            if not source.is_file():
                failures.append(f"Missing: {path}")
                continue
            raw = source.read_bytes()
            digest = lambda data: hashlib.sha256(data).hexdigest()
            if digest(raw) == entry["sha256"]:
                counts["exact"] += 1
            elif digest(raw.replace(b"\r\n", b"\n")) == entry["sha256"]:
                # Also verify committed bytes; do not mistake arbitrary edits for checkout conversion.
                blob = git("show", f"HEAD:{path}")
                if digest(blob) != entry["sha256"]:
                    failures.append(f"Committed hash mismatch: {path}")
                else:
                    counts["crlf_only"] += 1
            else:
                failures.append(f"Hash mismatch: {path}")
        result[label] = counts
    counts = {"python_files": 0, "notebook_cells": 0, "json_files": 0}
    for path in tracked:
        source = ROOT / path
        try:
            if path.endswith(".py"):
                ast.parse(source.read_text(encoding="utf-8-sig"), filename=path)
                counts["python_files"] += 1
            elif path.endswith((".json", ".ipynb")):
                value = json.loads(source.read_text(encoding="utf-8-sig"))
                if path.endswith(".ipynb"):
                    for cell in value.get("cells", []):
                        if cell.get("cell_type") == "code":
                            ast.parse("".join(cell["source"]), filename=path)
                            counts["notebook_cells"] += 1
                else:
                    counts["json_files"] += 1
        except (SyntaxError, ValueError, OSError) as error:
            failures.append(f"Parse error: {path}: {error}")
    result["syntax"] = counts
    result["yaml_validation"] = "Not performed: PyYAML is not required by this checker"
    result["failures"] = failures
    print(json.dumps(result, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
