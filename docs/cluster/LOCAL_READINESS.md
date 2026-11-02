# Local repository readiness — 2 November 2026

Scope: Phase A and the initial Phase B connectivity check. This is not a repeat
of the Phase 0 technical audit or a declaration of operational readiness.

## Baseline

- Windows checkout: `C:\Users\lakho\Desktop\NTU\FYP\JointPoseAction`.
- Initial branch: `main`; initially clean tracked/untracked working tree.
- Baseline commit: `5567cb1d6b6bb2b445a0fd3244dc7ee79c4b4b26`.
- Origin fetch/push: `https://github.com/shreyaslakhotia/JointPoseAction`.
- Read-only `git ls-remote origin refs/heads/main` returned this same commit.
- Migration work branch: `migration/phase1a-readiness`.
- Git runs with a command-scoped `safe.directory` exception because the sandbox
  identity differs from the checkout owner. No global Git setting was changed.

## Preserved source and exclusions

The migration manifest, seven Phase 0 reports plus inspection record, JSON
inventories, observed/restore environment files, upstream revision manifest,
SAM 3D custom source, vendor patch and nine checkpoint YAML snapshots exist.
The old restore instructions are historical recipes, not authorization to
install the old environments or all three upstream repositories.

Pinned upstream dependencies from `reproducibility/upstreams.json`:

| Dependency | Revision | Phase 1A handling |
|---|---|---|
| AssemblyHands toolkit | `8d4b7f1bce3f99874514199aae7504f78ebf3517` | Restore only when needed for metadata/calibration inspection |
| SAM 2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | Deferred; not needed for the environment smoke test |
| SAM 3D Objects | `f91db411c50efee93d8db7aeb323885650f6f722` | Preserve custom source/patch; do not rebuild the broken experiment |

None of these external checkouts is present at its recorded local restore
location. This is intentional exclusion, not evidence of lost custom code.
Raw videos, poses, model weights, environments and caches are ignored. An
ignored local `data/` directory exists; its presence does not establish a
downloaded or validated pilot. Historical image/CSV evidence remains in Git.
No essential custom source has been identified as missing within the manifest's
scope; undocumented work outside that scope cannot be certified.

## Windows/Linux boundaries

- `core.autocrlf=true`; checked source has LF in Git and CRLF in this Windows
  working tree. Transfer code by Git clone on Linux, not a Windows directory
  copy. Use committed bytes when checking preservation hashes. New shell scripts
  must have LF; invoke with `bash` or explicitly preserve executable bits.
- Reconstruction wrapper, data packager, SAM 3 mask defaults and SAM 2 checkpoint
  path contain `/home/mlda/...`. These are not portable runtime defaults.
- Existing renderers reuse one `9012-c07c` calibration. Do not run them against
  arbitrary recordings or interpret a successful import as verified calibration.
- Relative input paths assume execution from the repository root. External SAM 2
  is historically adjacent to the project; the toolkit/SAM 3D paths are nested.
- Old environments include workstation/CUDA assumptions; rebuild minimally after
  live GPU/module discovery. Conda activation is a Linux job concern.
- README references absent `download_subset.py` and `src/`; these are stale
  documentation, not a reason to invent replacement research code. The existing
  `scripts/download_dataset.py` has known source/selection defects documented in
  `docs/MIGRATION_MANIFEST.md`; do not run it for pilot acquisition.
- Case-sensitive Linux path resolution and actual GPU execution remain untested.

No historical research source or preservation hash was edited. No migration
blocker requires changing that source before read-only resource discovery.

## Repeatable checks

Run locally from the repository root with an available Python interpreter:

```powershell
python -B scripts/cluster/verify_checkout.py
```

The checker validates saved hashes, explicitly distinguishes CRLF-only checkout
conversion (also checking the committed blob), parses tracked Python/JSON and
notebook code without importing model code, and reports patch-only entries as
deferred. It does not install dependencies or prove algorithmic correctness.
Do not run repository-wide checks on cluster login nodes; use a CPU batch job
for cluster source verification. Results are in `local_validation.json`.

YAML parsing was separately checked with the existing local Anaconda Python and
PyYAML 6.0; no package installation was performed. Patch reconstruction against
upstream blobs was verified during preservation on 1 November and has not been
repeated here. Syntax checks do not establish runtime dependency availability.
