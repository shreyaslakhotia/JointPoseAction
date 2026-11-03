# File migration map

Baseline: `fcb0a40db35c1c20da76df5fa90fa7a9179d63ad`. All work is isolated on `chore/repository-hygiene`; production remains pinned.

The complete 710-file inventory, original Git blob IDs, canonical SHA-256, pre-move working-byte SHA-256, final paths and reasons are preserved in [repository_hygiene.json](../../reproducibility/manifests/repository_hygiene.json). No original tracked file is discarded.

## Significant move groups

| Original path | Preserved path | Files | Rationale |
|---|---|---:|---|
| `3D_reconstruction` | `legacy_files/historical_code/3D_reconstruction` | 2 | Historical reconstruction source; preserve unchanged for later geometry research |
| `experiments` | `legacy_files/old_experiments/experiments` | 215 | Historical SAM2/SAM3 exploratory scripts and lightweight artifacts; external active cluster runs are separate |
| `mistake_repo` | `legacy_files/old_experiments/mistake_repo` | 329 | Original data-only mistake annotation snapshots and README; not the recognition action-label source |
| `notebooks` | `legacy_files/historical_code/notebooks` | 1 | Historical pose visualization notebook |
| `reports` | `legacy_files/historical_reports/reports` | 15 | Dated Phase 0 audit records; preserve original findings |
| `docs/MIGRATION_MANIFEST.md` | `legacy_files/historical_reports/docs/MIGRATION_MANIFEST.md` | 1 | Preserve dated migration record; replace current path with a navigation wrapper |
| `docs/CLUSTER_RESTORE.md` | `legacy_files/historical_reports/docs/CLUSTER_RESTORE.md` | 1 | Preserve dated old-layout restoration recipe; replace current path with current restoration guidance |
| `scripts/download_dataset.py` | `legacy_files/deprecated_scripts/scripts/download_dataset.py` | 1 | Historical exploratory utility with workstation/old-layout assumptions; no active recognition imports |
| `scripts/extract_sam3_masks.py` | `legacy_files/deprecated_scripts/scripts/extract_sam3_masks.py` | 1 | Historical exploratory utility with workstation/old-layout assumptions; no active recognition imports |
| `scripts/package_reconstruction_data.py` | `legacy_files/deprecated_scripts/scripts/package_reconstruction_data.py` | 1 | Historical exploratory utility with workstation/old-layout assumptions; no active recognition imports |
| `scripts/render_all_views.py` | `legacy_files/deprecated_scripts/scripts/render_all_views.py` | 1 | Historical exploratory utility with workstation/old-layout assumptions; no active recognition imports |
| `scripts/render_batch.py` | `legacy_files/deprecated_scripts/scripts/render_batch.py` | 1 | Historical exploratory utility with workstation/old-layout assumptions; no active recognition imports |
| `scripts/visualize_single.py` | `legacy_files/deprecated_scripts/scripts/visualize_single.py` | 1 | Historical exploratory utility with workstation/old-layout assumptions; no active recognition imports |

## Preserved versions of files with current replacements

| Original path | Exact archived version |
|---|---|
| `.gitignore` | `legacy_files/historical_reports/root_gitignore_before_hygiene.txt` |
| `README.md` | `legacy_files/historical_reports/root_README_before_hygiene.md` |
| `scripts/cluster/verify_checkout.py` | `legacy_files/diagnostics/scripts/cluster/verify_checkout.py` |

Root README and ignore policy receive current replacements. The preservation checker receives path-only resolution support; its original implementation is archived. The two dated migration/restore guides are moved unchanged and replaced by short current navigation guides.

## Interfaces intentionally retained

`e0/`, `phase1c/`, all four configurations, all original tests, Slurm entry points, `scripts/data/`, the fixed dataset hash, upstream revisions/patches and dated reproducibility manifests retain their paths and bytes. The flat configuration and test layouts remain because the current CLI, tests and reproducibility recipes reference them. A `src/` packaging migration would add unnecessary compatibility changes and is deferred.

The historical tracked `experiments/` directory is separate from `$JPA_ROOT/experiments/e0` and `$JPA_ROOT/experiments/phase1c` on the cluster. No live result, checkpoint, cache, raw dataset, environment or cluster directory is moved.

Historical source retains original filenames and relationships; hardcoded workstation paths and old working-directory assumptions remain documented limitations. Original restore paths in dated manifests describe the original context. Use the path resolver for preservation verification, not an assumption that archived applications run unchanged from the new location.

No tracked original FYP proposal, PDF, Word or presentation file was found in the baseline. None was deleted or reconstructed.
