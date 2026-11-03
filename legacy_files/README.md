# Historical material

This archive preserves repository material outside the current E0/Phase 1C
recognition workflow. Archiving means preservation, not a judgment that the work
has no future value. Active recognition packages, configs, tests, data tools and
cluster scripts remain at their existing paths.

| Group | Contents and reason | Expected usability |
|---|---|---|
| `historical_code/3D_reconstruction/` | Earlier reconstruction prototypes and related code | Historical environments, models and external assets may be required; no current end-to-end validation is claimed |
| `historical_code/notebooks/` | Exploratory notebooks | Preserve outputs and original context; paths and kernels may need restoration |
| `old_experiments/experiments/` | Earlier experiment implementations and tracked artifacts | Read with accompanying reports/manifests; not current Phase 1C runtime outputs |
| `old_experiments/mistake_repo/` | Earlier mistake-detection exploration | Historical dependencies and dataset paths may be required |
| `deprecated_scripts/scripts/` | Original exploratory download, inspection and visualization entry points | Original filenames retained; original working directory and data assumptions may no longer resolve after relocation |
| `historical_reports/reports/` | Earlier technical, dataset, environment and component audits | Preserved evidence of its recorded phase, not assertions about current implementation |
| `historical_reports/root_README_before_hygiene.md` | Exact pre-hygiene root README, including historical setup and roadmap | Historical guidance; use the current root README for the recognition workflow |

Original directory relationships and file contents are preserved where practical.
See the complete [migration map](../docs/reproducibility/FILE_MIGRATION_MAP.md)
for original locations and the [hygiene report](../docs/reproducibility/REPOSITORY_HYGIENE_REPORT.md)
for byte-preservation checks and validation limitations.

Historical code is not claimed runnable from its new location. It may contain
relative paths, original repository-root assumptions or references to external
checkpoints. Recover the original layout from the migration map or Git history
in a separate sandbox before attempting a historical reproduction; do not
overwrite the active experiment checkout to do so. Content in historical reports
has not been revised to reflect later findings.

Large raw videos, pose archives, generated datasets, environments, upstream
checkouts and weights are not copied here. Their recorded paths, checksums,
acquisition sources and dependency relationships remain in the
[reproducibility manifests](../reproducibility/) and
[migration documentation](../docs/MIGRATION_MANIFEST.md). A tracked proposal or
authoritative research record, if present in an archived group, retains its
original bytes and research authority.
