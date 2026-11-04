# JointPoseAction

Can reconstructed 3D object geometry improve fine-grained assembly action
recognition beyond hand poses alone? JointPoseAction first establishes pose-only
TCN and HandFormer baselines on Assembly101, then plans controlled reconstruction
and geometry-fusion comparisons.

## Current research status

The frozen E0 TCN achieved **28.08% Top-1, 54.89% Top-5 and 9.22% macro-F1** on
9,760 held-out segments from a selected 100-class subset. Its
[audit](docs/experiments/E0_AUDIT.md) confirms the exported metrics and documents
strong frequent-class bias. These are subset results, not a full Assembly101
benchmark or participant-independent evaluation.

Phase 1C compares that frozen model with a balanced TCN recipe and the official
pose-only HandFormer architecture through a documented preprocessing adapter.
Latest supplied cluster status: preparation **243530 completed** (31 seconds),
balanced TCN **243531 completed** (17 minutes 19 seconds), HandFormer **243532
pending**, and final evaluation **243533 pending**. New held-out results are
therefore pending. This is a status snapshot, not live scheduler state.

Reconstruction and hand/object fusion remain future experiments. Historical
reconstruction prototypes are preserved separately; they are not validated
components of the current recognition pipeline.

## Repository guide

| Path | Purpose |
|---|---|
| `e0/` | Audited dataset, preprocessing, TCN and original baseline evaluation |
| `phase1c/` | Balanced TCN, official HandFormer adapter and matched comparison |
| `configs/` | Fixed dataset and training configurations |
| `scripts/cluster/`, `scripts/data/` | Cluster execution and dataset preparation |
| `tests/` | CPU unit and synthetic integration tests |
| `docs/` | [Documentation index](docs/README.md), protocols and execution guides |
| `reproducibility/` | Environment snapshots, upstream pins, patches and evidence |
| `legacy_files/` | [Preserved historical code and reports](legacy_files/README.md) |

Active package paths are intentionally retained to protect imports, checkpoints
and existing experiment provenance. See the
[migration map](docs/reproducibility/FILE_MIGRATION_MAP.md) and
[hygiene report](docs/reproducibility/REPOSITORY_HYGIENE_REPORT.md).

## Setup, data and execution

Use the dedicated `/projects/jointposeaction` cluster allocation. Environment
setup and interpreter selection are documented in
[ENVIRONMENT_SETUP.md](docs/cluster/ENVIRONMENT_SETUP.md); current jobs use the
existing project Python environment rather than the historical root setup.

Read the [E0 dataset contract](docs/experiments/E0_DATASET.md) and
[schema validation](docs/experiments/E0_SCHEMA_VALIDATION.md) before preparing
data. The workflow uses official fine-grained action labels and recording-disjoint
splits; participants overlap between splits, and overlapping action intervals
remain separate examples. Large datasets, weights and caches stay outside Git.
Authenticate privately for gated access and use selective acquisition, not a full
RGB or pose-archive download.

The [Phase 1C execution guide](docs/cluster/PHASE1C_EXECUTION.md) describes the
submission entry point, preparation and sanity gates, training, final evaluation,
and recovery procedure. **Do not resubmit the existing workflow or deploy this
hygiene branch while its cluster jobs are pending or running.** Those jobs remain
pinned to `fcb0a40db35c1c20da76df5fa90fa7a9179d63ad`. The hygiene branch needs
review and explicit merge approval before any later deployment.

Run the CPU test suite in a compatible environment with:

```bash
python -m unittest discover -s tests -v
```

Runtime results remain outside the repository under
`/projects/jointposeaction/experiments/e0/controlled-baseline/` and
`/projects/jointposeaction/experiments/phase1c/recognition-baselines/`.
The latter's `comparison/` directory receives final model reports and comparisons
after both new training jobs succeed. Tracked experiment documents distinguish
recipes, observed results and pending measurements.

For the future geometry work, start with
[fusion readiness](docs/experiments/FUSION_READINESS.md). Absolute video/pose clock
alignment and metric hand/object calibration remain explicit validation gates.
