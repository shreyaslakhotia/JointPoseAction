# Project scope and navigation

The research question is whether reconstructed 3D object geometry improves
fine-grained assembly action recognition beyond hand poses. The current stage
establishes pose-only recognition controls on a fixed Assembly101 subset before
introducing geometry.

Three controls are kept distinct: frozen E0/B0-original TCN; B0-balanced with
class weighting and validation macro-F1 selection; and B1 using the official
pose-only HandFormer architecture with explicitly documented preprocessing
differences. The comparison does not isolate architecture alone, and the
balanced recipe changes both weighting and checkpoint selection.

The official recording-disjoint protocol permits participants to overlap across
splits. Vocabulary selection uses training data only. Fine-grained intervals can
overlap and are preserved as distinct examples. Absolute video/pose alignment
and metric hand/object calibration are not established by the pose-only results.

Use the [documentation index](../README.md) for protocols and execution details,
the [root README](../../README.md) for the latest supplied status, and
[fusion readiness](../experiments/FUSION_READINESS.md) for the later geometry
interface and validation gates.

The hygiene branch reorganizes preservation and navigation only. Existing
cluster jobs use their pinned production revision; do not deploy the hygiene
branch to those jobs. Existing research configurations, packages, datasets and
checkpoints retain their original interfaces.
