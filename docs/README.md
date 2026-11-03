# Documentation index

Start with the [project overview](project/README.md). Phase-specific documents
retain their recorded findings and execution state; older setup notes are not
live evidence of the latest cluster status.

## Active recognition work

| Topic | Canonical entry point |
|---|---|
| Current experiment execution | [Phase 1C cluster workflow](cluster/PHASE1C_EXECUTION.md) |
| Environment and absolute interpreter | [Environment setup](cluster/ENVIRONMENT_SETUP.md) |
| Source annotations and selection | [E0 dataset contract](experiments/E0_DATASET.md) |
| Pose format, masks and nominal alignment | [E0 schema validation](experiments/E0_SCHEMA_VALIDATION.md) |
| Original TCN implementation | [E0 implementation](experiments/E0_IMPLEMENTATION.md) |
| Original measured results | [E0 results](experiments/E0_RESULTS.md) |
| Independent results investigation | [E0 audit](experiments/E0_AUDIT.md) |
| Balanced TCN recipe and reporting | [B0-balanced](experiments/B0_BALANCED_RESULTS.md) |
| Official pose-only HandFormer adaptation | [B1-HandFormer](experiments/B1_HANDFORMER_RESULTS.md) |
| Matched population and comparison | [Dual-backbone comparison](experiments/DUAL_BACKBONE_COMPARISON.md) |
| Future reconstruction/fusion gates | [Fusion readiness](experiments/FUSION_READINESS.md) |

## Preservation and provenance

- [File migration map](reproducibility/FILE_MIGRATION_MAP.md) and
  [repository hygiene report](reproducibility/REPOSITORY_HYGIENE_REPORT.md) explain
  the isolated reorganization and validation.
- [Reproducibility resources](../reproducibility/) retain dataset/upstream pins,
  patches, environment snapshots and machine-readable experiment evidence.
- [Migration manifest](MIGRATION_MANIFEST.md) and
  [cluster restore](CLUSTER_RESTORE.md) preserve workstation migration history.
  Their original paths and earlier defects are historical provenance, not a
  replacement for the current recognition protocol.
- [Historical archive](../legacy_files/README.md) locates prior experiments,
  notebooks, Phase 0 audit reports, and the former root README. Historical reports
  keep their original findings; they are not rewritten as current results.
- `data/` documentation records early source discovery and pilot planning.
  For the acquired E0 population, prefer the experiment dataset/schema documents
  and pinned manifests above.

The latest supplied job state appears in the root README. The generated
cluster run directories and scheduler records are authoritative for subsequent
completion, checkpoints and measured results.
