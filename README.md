# JointPoseAction

Can reconstructed 3D object geometry improve fine-grained assembly action
recognition beyond hand poses alone? The project has completed pose-only TCN
and HandFormer controls on Assembly101. Reconstruction, calibration and
hand-object fusion are the next research stages; they are not implemented or
validated by these recognition results.

## Segmentation and reconstruction preparation

The [research-artifact index](research_artifacts/README.md) records three
diagnostic Assembly101 segmentation candidates. SAM 3 inference job `270796`
and CPU review job `271689` completed. The
[inspection report](research_artifacts/segmentation_pilot/INSPECTION_270796.md)
preserves successful predictions and background/tool failures. Camera accuracy,
temporal correspondence and final mask approval remain unresolved; no pilot is
accepted for reconstruction yet.

The new [reconstruction implementation](research_artifacts/reconstruction_benchmark/READINESS.md)
is validated on synthetic primitives only. It provides calibrated voxel carving,
solid occupancy rendering, point-cloud export, held-out-camera evaluation and
metadata adapters for future learned methods. Real-object reconstruction and
additional GPU model inference are disabled in this phase.

## Measured recognition results

All three frozen checkpoints were evaluated on the same **9,760 segments from
67 recordings**, covering 100 classes. The dataset contains 22,083 training and
6,880 validation segments. Official recording-disjoint splits are retained;
participants overlap across splits. These are selected-subset results, not a
full Assembly101 benchmark or participant-independent evaluation.

| Model | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| Original TCN | 28.08% | 54.89% | 9.22% |
| Balanced TCN | 21.53% | 49.26% | 11.44% |
| HandFormer pose-only adaptation | 36.31% | 65.92% | 17.09% |

HandFormer is the strongest measured recognizer. The balanced TCN broadens
class coverage at a substantial accuracy cost. All three are retained for
future within-backbone comparisons; differences also include preprocessing,
loss and checkpoint selection, so they do not isolate architecture alone.
Read the [final comparison](docs/experiments/DUAL_BACKBONE_COMPARISON.md),
[B0 report](docs/experiments/B0_BALANCED_RESULTS.md),
[B1 report](docs/experiments/B1_HANDFORMER_RESULTS.md) and
[original audit](docs/experiments/E0_AUDIT.md).

## Repository structure

```text
JointPoseAction/
|-- e0/                 # Shared dataset/feature tools, original TCN and audit
|-- phase1c/            # Balanced TCN, official HandFormer adapter, comparison
|-- segmentation_pilot/ # Bounded acquisition, calibration diagnostics, native SAM 3
|-- reconstruction/     # CPU geometry, evaluation, metadata-only model adapters
|-- configs/            # Frozen dataset and experiment configurations
|-- scripts/
|   |-- cluster/        # Environment checks, Slurm and submission wrappers
|   |-- data/           # Selective acquisition and dataset validation
|   `-- reproducibility/ # Read-only repository preservation validation
|-- tests/              # CPU regression, synthetic-data and model tests
|-- docs/               # Current results, protocols, setup and migration map
|-- reproducibility/    # Pins, environments, patches, manifests and evidence
|-- research_artifacts/ # Attributed review figures and synthetic validation
`-- legacy_files/       # Historical reconstruction, notebooks and artifacts
```

`e0/` and `phase1c/` are the active recognition packages. Their module names,
imports and CLI entry points are retained to preserve tested experiment
interfaces. The geometry package is independently tested and does not alter
these recognition interfaces or enable real reconstruction experiments.
See the [documentation index](docs/README.md),
[archive inventory](legacy_files/README.md),
[file migration map](docs/reproducibility/FILE_MIGRATION_MAP.md) and
[consolidation report](docs/reproducibility/REPOSITORY_HYGIENE_REPORT.md).
`main` is the authoritative source branch. Historical experiment source
revisions remain recoverable through Git history after branch cleanup.

## Setup and validation

Use the dedicated `/projects/jointposeaction` allocation. The
[environment guide](docs/cluster/ENVIRONMENT_SETUP.md) documents the restored
project interpreter and pinned dependencies. Local CPU validation uses an
existing compatible Python environment; it does not require downloading data
or training a model.

```bash
python -m unittest discover -s tests -v
python scripts/reproducibility/validate_hygiene.py
python scripts/cluster/verify_checkout.py
python -m phase1c.train --help
python -m phase1c.evaluate --help
python -m phase1c.compare --help
```

For the official HandFormer forward/backward CPU test, set
`JPA_HANDFORMER_ROOT` to an existing checkout of upstream revision
`640a10a6bd03f85842a2662e67cf30444ab1d673`. Its source is verified against
[canonical fingerprints](reproducibility/upstreams/handformer.json).
The environment requirements and isolated source adapter are documented in
the B1 report. Do not commit upstream clones, environments or tokens.

## Data and reproduction

Read the [dataset contract](docs/experiments/E0_DATASET.md) and
[pose schema/alignment record](docs/experiments/E0_SCHEMA_VALIDATION.md).
Vocabulary selection uses training data only; overlapping action intervals
remain separate examples. Positive source confidence and finite coordinates
define validity, and valid zero coordinates are preserved. Features are
dimensionless. Absolute pose/video synchronization and metric hand/object
calibration remain unverified.

Frozen dataset SHA-256:
`f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`.
Original TCN training source: `b188995a566e74de9d10f40baf0884319f2e576c`.
Balanced TCN/HandFormer training source:
`fcb0a40db35c1c20da76df5fa90fa7a9179d63ad`.
Completed results and history snapshots are recorded in
[phase1c_completed_243533.json](reproducibility/experiments/phase1c_completed_243533.json).

The [execution guide](docs/cluster/PHASE1C_EXECUTION.md) records configurations,
submission procedures, validation gates and source pins. For exact reproduction,
use the recorded experiment revision in a separate checkout, retain the frozen
dataset/configuration, and configure a new run destination. Existing pipeline
state and outputs must not be reused or overwritten. Do not run the submission
wrapper merely to inspect the completed experiments.

Large datasets, checkpoints, histories and prediction arrays remain outside Git:

- Original TCN: `/projects/jointposeaction/experiments/e0/controlled-baseline/`.
- Phase 1C: `/projects/jointposeaction/experiments/phase1c/recognition-baselines/`.
  Selected checkpoints: `b0/train/best.pt` and `b1/train/best.pt`;
  reports: `comparison/`; histories: each model's `train/history.jsonl`.

This consolidation does not alter or deploy to the cluster checkout or change
its datasets, checkpoints or run artifacts. Local ignored videos, caches and
source-review checkouts are retained outside version control. Historical
reconstruction prototypes are preserved in `legacy_files/`; their dependencies
and original path assumptions require deliberate restoration.

For the next research phase, begin with
[fusion readiness](docs/experiments/FUSION_READINESS.md). Preserve the frozen
recognition controls and validate geometry/alignment before any fusion claim.
