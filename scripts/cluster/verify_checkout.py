"""Read-only, dependency-free preservation checks. Run locally, not on login nodes."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
        stderr=subprocess.PIPE,
    )


def migration_paths():
    """Resolve current paths without rewriting dated preservation manifests."""
    manifest = ROOT / "reproducibility/manifests/repository_hygiene.json"
    if not manifest.is_file():
        return {}
    mappings = {}
    for row in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
        original, final = row["original_path"], row["final_path"]
        for value in (original, final):
            candidate = Path(value)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"Unsafe preservation path: {value}")
        if original in mappings:
            raise ValueError(f"Duplicate preservation mapping: {original}")
        mappings[original] = final
    return mappings


def resolve_preserved_path(original, mappings):
    path = mappings.get(original, original)
    if not (ROOT / path).is_file():
        raise FileNotFoundError(f"Missing: {original} (resolved: {path})")
    return path


def verify_preserved(original, expected, mappings):
    path = resolve_preserved_path(original, mappings)
    raw = (ROOT / path).read_bytes()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    if digest(raw) == expected:
        return "exact"
    if digest(raw.replace(b"\r\n", b"\n")) == expected:
        # Before the hygiene commit, HEAD still names the original location.
        # A present new blob always wins; a hash mismatch must never fall back.
        try:
            blob = git("show", f"HEAD:{path}")
        except subprocess.CalledProcessError:
            if path == original:
                raise
            blob = git("show", f"HEAD:{original}")
        if digest(blob) != expected:
            raise ValueError(f"Committed hash mismatch: {path}")
        return "crlf_only"
    raise ValueError(f"Hash mismatch: {path}")


def main():
    tracked = git("ls-files", "-z").decode().strip("\0").split("\0")
    result = {"commit": git("rev-parse", "HEAD").decode().strip(),
              "preservation": {}, "small_artifacts": {}, "syntax": {},
              "patch_reconstruction": "Deferred: requires pinned upstream blobs"}
    failures = []
    mappings = migration_paths()
    for label, manifest, key in (
        ("preservation", "preservation_checksums.json", "preserved_as"),
        ("small_artifacts", "small_artifacts.json", "path"),
    ):
        counts = {"exact": 0, "crlf_only": 0, "deferred_patch_entries": 0}
        for entry in json.loads((ROOT / "reproducibility" / manifest).read_text())["files"]:
            if key not in entry:
                counts["deferred_patch_entries"] += 1
                continue
            try:
                counts[verify_preserved(entry[key], entry["sha256"], mappings)] += 1
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                failures.append(str(error))
        result[label] = counts
    counts = {"python_files": 0, "notebook_cells": 0, "json_files": 0}
    for path in tracked:
        try:
            # Updated entry points can coexist with archived original snapshots.
            # Syntax validation covers the current tracked file at its own path;
            # only pre-commit moves with an old index need relocation fallback.
            source = ROOT / path
            if not source.is_file():
                source = ROOT / resolve_preserved_path(path, mappings)
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
