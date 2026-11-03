# Cluster restoration and historical layouts

For the active recognition environment, use
[environment setup](cluster/ENVIRONMENT_SETUP.md), the
[Phase 1C workflow](cluster/PHASE1C_EXECUTION.md), and the
[dataset contract](experiments/E0_DATASET.md). These retain their existing module,
configuration, dataset and checkpoint interfaces.

The original workstation restoration guide is preserved unchanged in
[the historical archive](../legacy_files/historical_reports/docs/CLUSTER_RESTORE.md).
It describes the earlier directory layout, environment snapshots, upstream
patches, and pre-existing defects. Commands and absolute paths inside that dated
document belong to its original context; they are not instructions to overwrite
the current recognition workspace.

For exact old-layout reproduction, inspect production revision
`fcb0a40db35c1c20da76df5fa90fa7a9179d63ad` or the earlier commit recorded by the
historical experiment. For locating preserved material in this hygiene branch,
use the [file migration map](reproducibility/FILE_MIGRATION_MAP.md). Original
paths inside dated manifests remain intact; `scripts/cluster/verify_checkout.py`
resolves them through the new machine-readable migration map.

Large data, checkpoints, caches, environments and external Git clones stay outside
Git. The retained [upstream manifest](../reproducibility/upstreams.json) and
[environment snapshots](../reproducibility/environments/) record their identities
and original restoration locations, rather than asserting that all historical
applications run from their new archive paths.

The hygiene branch must remain separate while the Phase 1C jobs use their pinned
production checkout. Merge and cluster deployment require a later review after
those results are secured.
