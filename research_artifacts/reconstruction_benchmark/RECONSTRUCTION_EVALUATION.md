# Reconstruction evaluation contract

This is an implementation contract, not a report of real reconstruction results.
Real-data execution remains blocked until the root readiness checks independently
establish acceptable calibration, temporal correspondence, object state, and masks.
Synthetic unit tests do not clear those gates.

## Fixed camera partition and observation provenance

`reconstruction.evaluation.ViewPartition` records the input cameras, held-out
cameras, and alignment cameras before scoring. Input and held-out lists must be
nonempty, disjoint, and partition the expected cameras. The default expected list
contains the eight fixed Assembly101 cameras. A deliberately reduced synthetic
fixture may explicitly supply another expected list.

The reconstructed artifact must record the actual construction cameras in
`metadata.retained_camera_ids` (visual hull) or `metadata.input_camera_ids`
(learned surface). These must be a nonempty subset of the fixed input list.
`metadata.alignment_camera_ids` must equal the declared alignment list and contain
input cameras only. Missing provenance or a held-out camera used for construction
or alignment raises an error. A skipped input mask can reduce the actual input
list, but cannot silently replace it with a held-out view.

Each `Observation` contains the calibrated camera, native-resolution binary mask,
SHA-256 of its source mask artifact, reference kind, and provenance. A reference
kind of `predicted` produces **predicted-mask agreement**, even if agreement is
perfect. It does not establish independent reconstruction accuracy. Claiming
`independent_verified` requires independently established manual/dataset annotation
provenance: `verified_independent=true`, `derived_from_prediction=false`, annotation
kind, source ID, reviewer, and evidence. These are review attestations, not proof
inferred by software from a model's own output.

## Silhouette scores and missing views

`evaluate_silhouettes` uses the shared native-resolution IoU, Dice, and boundary-F1
implementations in `segmentation_pilot.metrics`. Boundary tolerance is specified in
original pixels (default 2). Both-empty masks score 1; exactly one empty scores 0.
Per-view foreground counts expose this convention. No resizing is implicit.

The default renderer traces occupied voxel boxes at pixel centers. It evaluates a
volume silhouette rather than the projection of voxel centers. Its discretization
and image sampling can differ from the center-based carving approximation. A
future mesh can use an explicit rasterizer hook, with finite vertices, valid
integer triangle connectivity, and declared common camera frame/units. There is no
default mesh rasterizer. A point cloud cannot supply filled silhouettes: scattered
point visibility is separately named `point_projection_coverage_not_silhouette`.

Every expected camera receives a report row. Missing masks remain `missing`, with
null scores, and are listed explicitly. Means are arithmetic means over available
cameras, separately for input versus held-out cameras and predicted versus verified
references. A result on five available cameras is never divided by eight or
described as an eight-view measurement. No available cameras produces a null mean.

## Alignment and temporal geometry

`fit_landmark_similarity` estimates a proper rotation, finite positive scale, and
translation from at least three non-collinear, independently paired 3-D landmarks.
Rows must have a documented physical correspondence. The returned transform records
source/target frames, source/target units, method, and landmark evidence. Landmark
fit residuals measure in-sample fit; they do not establish held-out alignment
accuracy. Reflections, coincident/collinear landmarks, arbitrary box normalization,
and unrecorded units are rejected. The function does not infer that canonical
single-image model coordinates have physical metric meaning.

Any later mask-based alignment must record optimization views and use input
cameras only. The framework does not implement mask-based fitting or use held-out
masks to select a transform. Bounding-box agreement is not a calibration test.

`temporal_chamfer` reports the average of the two directed mean Euclidean
nearest-neighbor distances. Inputs must already share explicitly named frame and
units and carry applied registration metadata. It does not register shapes,
establish semantic correspondences, preserve point identities, or measure motion
accuracy. Permuting point order does not change this shape diagnostic. The CPU
implementation rejects more than ten million point pairs or ten thousand points
per shape; it never silently subsamples. A parameter can lower the pair budget.

## Resource and visualization interfaces

`runtime_record` records elapsed time, output bytes, status, failure reason, and
peak-memory bytes with measurement method. Unknown peak memory stays null. Failed
or blocked runs require a reason. CPU process memory and CUDA allocator memory
must be identified distinctly by the caller rather than conflated.

`figures.silhouette_overlay` uses cyan reference and orange rendered contours;
white indicates coincident contours. It preserves native RGB resolution and saves
a visualization only, never an input mask. `plot_camera_metrics` plots supplied
per-camera scores and labels missing views. `comparison_markdown` separates
agreement from independent-reference scores and prints unavailable means honestly.
No figure utility generates measurements or claims real experimental results.

## CPU verification

Run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_reconstruction_evaluation.py' -v
```

The synthetic tests cover exact similarity recovery; reflection and collinearity
rejection; partition and actual artifact provenance leakage; missing-view
aggregation; native shape validation; predicted versus independent reference
labels; mesh connectivity versus point scatter; point visibility; registered
Chamfer with known distances and permuted point order; CPU bounds; resource
records; and native-resolution overlays. They require no model, trained checkpoint,
dataset download, cluster job, or GPU.
