# Future geometric fusion interfaces

**Scope: interface planning only. No reconstruction, object tracking, object encoder or fusion module is implemented in Phase 1C.** The pose-only runs establish reference models for a later matched comparison.

## Shared example and time contract

Every future object-feature record must identify the frozen source dataset hash, `segment_id`, `recording_id`, official split and selected action label. Preserve overlapping fine-grained action intervals as separate segment-level examples. Use the existing training-only vocabulary, official recording-disjoint split assignments and a deterministically defined evaluation population. Participants overlap across splits.

Store explicit source frame identities and timestamps alongside object descriptors and the pose sampling indices. The recognition contract currently interprets annotation intervals at 30 FPS and pose indices at nominal 60 FPS, with zero-based indexing and the documented inclusive annotation endpoint conversion. Do not infer correspondence from the length of a tensor. Source confidence and per-object/per-frame validity must be explicit; a valid zero coordinate must not silently become a missing observation.

Absolute pose-to-video synchronization is not independently verified. Recognition's dimensionless hand normalization also does not establish metric calibration between a reconstructed object and the hand. Before claiming metric distances, contact or relative orientation, the geometry phase must validate clock correspondence, coordinate transforms, units, axis conventions and calibration. Those validations are later research gates, not implicit consequences of these recognition baselines.

## TCN interface

The frozen pose-only TCN consumes `float32 [batch, 64, 216]` features; its forward path transposes to channels-first for temporal convolution. The 216 channels comprise 126 hand-local joint coordinates, six common-origin wrist/root coordinates, 42 raw joint-validity flags and 42 normalized joint-validity flags. Joint order, left/right order and normalization are fixed by the E0 feature contract.

A future object descriptor may use `[batch, 64, object_feature_dim]`, with a separate `[batch, 64, object_count]` visibility/validity mask where relevant. A descriptor encoder could map these features to a controlled channel width before concatenation with the hand channels or before fusion with the TCN temporal embedding. Choosing an insertion point changes the model and must be documented. Keep the original pose-only representation available; do not add semantic action labels or ground-truth object identity as a shortcut unless that supervision is explicitly part of the declared experiment.

Metric reconstructed descriptors and dimensionless hand coordinates cannot be concatenated without a deliberate normalization contract. Candidate options include geometry expressed relative to the same wrist origin and hand-length scale, or separately encoded metric geometry with documented calibration. Object tracking identities, variable object counts and confidence propagation need an explicit policy before implementation.

## HandFormer interface

HandFormer uses the official pose-only temporal/micro-action pathway through a compatibility adapter. Its native joint tensor is `float32 [batch, 3, 120, 21, 2]` (coordinates, sampled time, joints, hands). The hand axis is left then right. Source frames retain world-axis orientation and use the E0 median wrist-to-index-base bone scale and segment origin; nonfinite observations and source confidence at or below zero become zero inputs, while finite zero coordinates with positive confidence remain valid. Its temporal sampling and tensor representation differ from the custom TCN's 216-channel features. Do not feed flattened TCN features into the official pose encoder or replace its micro-action representation to simplify fusion.

The adapter resizes temporal crops to 120 steps using explicit sequence length, without treating coordinate sums as a missing-frame detector. Training crops span 70–100% of the segment with a 60-frame minimum; evaluation uses the center 85%. The integration adds no extra annotation context. These preprocessing choices must be retained or controlled when later introducing geometry.

The configured pose pathway forms 15-step micro-action windows at stride eight, yielding 14 windows from 120 steps and a 256-dimensional token per micro-action. A separate global-wrist temporal pathway supplies features to the trajectory encoder. The hand encoders share weights and their outputs are averaged across hands. The temporal classifier consumes the micro-action tokens using a two-layer, eight-head transformer and a 256-dimensional classification token before the 100-class head.

The future geometry branch should retain the source timestamp/frame identity for every HandFormer sample and micro-action window. A candidate object encoder could produce temporally matched 256-dimensional tokens for a controlled fusion mechanism before the pose temporal transformer. This is an insertion candidate, not an implemented or validated design. The attention interface, object count handling and handling of object absence remain design choices for the geometry phase. The published pose+RGB path is not evidence that arbitrary metric object tokens can be substituted without checking shape, normalization and attention semantics.

The exact adapter tensor shape and token insertion dimensions must be taken from the Phase 1C integration and its recorded model configuration. Future code should assert those values from the pinned implementation instead of assuming that a different upstream revision has the same interface. A geometry-enabled model must preserve a separately runnable matched pose-only control.

## Fair pose-only versus fusion comparisons

Freeze examples, action vocabulary, splits, pose observations, checkpoint-selection criterion and evaluation metrics before comparing each architecture with its geometry extension. Train geometry transforms, normalizers and any learned encoders exclusively on training data. Use validation data for model selection and keep test data out of feature-policy decisions. If unavailable object tracks would remove examples, establish an explicit missing-feature strategy or a separately reported common population for both pose-only and fusion checkpoints.

Report each model family's change separately before comparing families. Document extra parameters, preprocessing, training budget, supervision and input observations. Use the existing prediction export and recording-cluster paired comparison so overlapping action segments are not treated as independent observations. A recognition gain alone does not establish a metric reconstruction improvement or its causal mechanism; later controlled ablations must support that explanation.
