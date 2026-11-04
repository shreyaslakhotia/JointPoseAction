# B1: official HandFormer pose-only adaptation

Status: training job 243532 and evaluation/comparison job 243533 completed
successfully. The user supplied the generated cluster comparison on 4 November
2026; the measurements below describe the fixed 100-class protocol.

## Measured outcome

All models use the same 9,760 held-out segments from 67 recordings.

| Model | Top-1 | Top-5 | Macro-F1 | Weighted-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| B0-original | 28.08% | 54.89% | 9.22% | 22.47% | 10.44% |
| B1-HandFormer | 36.31% | 65.92% | 17.09% | 32.65% | 17.95% |

HandFormer improves Top-1 by 8.23 percentage points and macro-F1 by 7.87 points
over the original TCN. The paired recording-cluster 95% interval for its Top-1
gain is [+6.94, +9.58] points. Against B0-balanced, the Top-1 gain is 14.78 points
with interval [+13.51, +16.04]. These intervals condition on the trained models
and selected recordings; they do not quantify seed variability or architecture's
isolated causal effect.

Fourteen classes have zero correct predictions and five are never predicted,
compared with 45 and 19 for the original TCN. Top-ten prediction concentration
is 59.52%. Verb/noun accuracy is 61.08%/48.52%. HandFormer gives the strongest
overall metrics of these runs, while B0-balanced reaches more classes at least
once (92 versus 86). Macro-F1 of 17.09% still indicates weak average class-level
performance, despite the improvement.

Validation macro-F1 selected epoch 41. The model has 3,805,866 parameters;
fitting time is 14,696.75 seconds (about 4h05m), and total GPU job elapsed is
4h13m34s including sanity/startup. GPU: NVIDIA L40S; peak PyTorch allocated CUDA
memory: 8,716,908,032 bytes (about 8.12 GiB). These are reported runtime measures,
not an estimate from parameter count. The TCN has substantially lower resource
cost.

The user also supplied first/selected/last history snapshots: all 50 epochs ran,
and the training-only sanity gate passed after 50 updates (7.87 seconds).
At epoch 41, online training Top-1/macro-F1 are 49.65%/29.67%, validation is
35.36%/16.01%, and test is 36.31%/17.09%. At epoch 50, online training improves
to 51.87%/31.93%, but validation falls to 34.45%/15.42%; validation CE rises
from 2.731 to 2.829. This pattern is consistent with late overfitting and
supports keeping the validation-selected epoch 41. Online training metrics use
dropout, augmentation and changing parameters, so their gap from validation is
not an exact post-fit generalization-gap estimate. The snapshots do not establish
the dominant cause of remaining errors, nor do they motivate automatic retraining.

Selected checkpoint SHA-256:
`6249b77c877b5d43f1e953721746fb89dfd82647184315131443297c1635d643`.

## Source and model

The [official implementation](https://github.com/s-shamil/HandFormer/tree/640a10a6bd03f85842a2662e67cf30444ab1d673)
is pinned to `640a10a6bd03f85842a2662e67cf30444ab1d673`. The seven executed
source files have SHA-256 hashes in
`reproducibility/upstreams/handformer.json`. The checkout lives outside the
project repository at `$JPA_ROOT/upstreams/HandFormer`; it contains source only,
without pretrained weights, RGB features, or dataset downloads.

Fingerprints refer to the canonical bytes obtained from
`git show <pinned-revision>:<path>`, which use LF in this revision. Verification
checks those Git blobs independently and compares each working file after only
CRLF-to-LF normalization. Windows checkout newline conversion is accepted;
semantic changes, whitespace changes and other edits are rejected. This fixes
the first cluster preparation attempt, which completed the 3.70 GB compact cache
but stopped because the initial fingerprints described Windows CRLF checkout
bytes instead of canonical Git blobs. The cached observations remain reusable;
this failure occurred before either model trained.

`phase1c/handformer.py` loads upstream `HF_Pose` and its actual component classes.
The [commented pose-only model block in the released training YAML](https://github.com/s-shamil/HandFormer/blob/640a10a6bd03f85842a2662e67cf30444ab1d673/HandFormer/train_config.yaml)
specifies 21 joints, 256-dimensional micro-action tokens, 15-frame windows with
configured overlap 0.5, shared hand encoders, the global wrist trajectory and
orientation reference, and a two-layer, eight-head transformer. Those settings
are preserved. The classifier has 100 outputs, totaling **3,805,866 parameters**.
No pretrained checkpoint or RGB branch is used.

The input is `batch × 3 × 120 × 21 × 2`, with hand order left/right and original
Assembly101 joint order (wrist 5, middle-proximal joint 11). The released integer
window-stride rule gives stride 8 and 14 micro-action tokens for 120 steps.

## Frozen population and source observations

The exact E0 dataset SHA-256 is
`f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`.
Expected counts remain train 22,083, validation 6,880 and test 9,760, across the
same 100 labels. The adapter verifies the ordered segment IDs and label mapping;
it does not select classes or silently discard examples. Intervals outside source
bounds stop preparation before training. Internally missing nominal frame IDs
remain explicit zero-filled positions; per-recording and per-segment counts,
including missing positions among the original 64 E0 samples, are reported.

The compact cache reads the existing cleaned pose/confidence files, one recording
at a time, verifies source hashes, and stores source frame IDs, float32 XYZ and
Boolean joint-validity arrays. It does not save repeated overlapping segments or
resize poses before training augmentation. Cache bytes are measured before each
recording is written and capped at 6 GB, with a further 2 GB of project allocation
headroom. A hash-verified per-recording progress journal allows safe resumption
without deleting existing data. The compatibility manifest records actual sizes and hashes. No pose,
transform or RGB payload is downloaded.

Pose bounds stay `[2*start_frame, 2*(end_frame+1))`. All available pose observations
inside the same action interval are eligible, while the original TCN sampled 64
nearest frames. There is no extra temporal context. Recording splits remain
official and disjoint; participants overlap across splits.

## Explicit differences from authored preprocessing

This is an official-architecture adaptation to the audited subset, not a numerical
replication of the authors' full Assembly101 experiment.

- The author's [clip reader](https://github.com/s-shamil/HandFormer/blob/640a10a6bd03f85842a2662e67cf30444ab1d673/Preprocess/1_get_clip_data.py)
  defaults to 30 annotation frames of context and clamps the start to frame 1.
  We preserve the frozen E0 interval and valid frame 0, without context expansion.
- The authored preprocessing applies a smoothed left-wrist transformation using
  `xf_transf` and PyTorch3D. We reuse each frozen E0 segment's shared origin and
  median wrist-to-index-proximal bone scale, preserving release world axes.
  This preserves relative wrist positions and motion without fetching additional
  transforms or claiming verified metric units.
- Nonfinite joints and hands with source confidence <=0 are zero-filled. Finite
  zero coordinates with positive confidence are preserved as observations. No
  confidence channel or extra learned masking component is added to HF_Pose.
- The authored temporal resize infers length from nonzero coordinate sums and can
  truncate valid zero-valued frames. Our small adapter uses the explicit selected
  length. It retains the same bilinear temporal interpolation with
  `align_corners=False`, including interpolation of zero-filled missing inputs.

Training follows the released temporal crop interval `[0.70,1.0]`, with the same
minimum crop-length rule of 60 source frames. Validation and test take the
deterministic central fraction 0.85. Crop precedes resize to 120 frames. Sanity
data use deterministic preprocessing.

Absolute pose/video synchronization remains independently unverified. The E0
audit's retained-recording cadence checks support the nominal temporal contract,
not an absolute RGB alignment or physical calibration claim. A B0/B1 difference
therefore reflects representation, preprocessing and architecture together.

## Compatibility transformations

Upstream files remain unchanged on disk. A hash-guarded loader applies only the
following executable compatibility transformations:

1. Isolate absolute `models.*`/`utils` imports in a private namespace, avoiding
   the upstream package initializer's unnecessary RGB imports.
2. Omit demo/profiling `__main__` blocks, `sys.path.insert` side effects, and their
   `thop`, `fvcore`, and `tqdm` imports.
3. Change the exact `batch_seq_ain_pe.get_device()` use to `.device`, which allows
   CPU checks while preserving CUDA device placement.
4. Replace the sole known `einops.repeat` action-token broadcast with tensor
   `expand(batch_size,-1,-1)`. An explicit value and gradient parity test covers
   this change. Unknown broadcast patterns are rejected.

No core network computation, parameter shape, attention rule or convolution is
rewritten. Existing NumPy and PyTorch are sufficient for this adapter; no
historical HandFormer environment is installed.

## Training protocol

`configs/b1_handformer.json` fixes seed 42, batch 32, SGD learning rate 0.025,
momentum 0.9 with Nesterov, weight decay 0.0005, five warmup epochs, learning-rate
decays at zero-indexed epochs 25/40 by 0.1, and 50 epochs. These optimizer and
objective settings follow the released YAML; label smoothing is 0.1 and logits
are divided by temperature 1.1 for the optimization loss. No auxiliary RGB,
feature, verb or noun loss is used by the pose-only head.

Validation macro-F1 selects the checkpoint for comparison with B0-balanced;
validation Top-1 and plain cross-entropy are reported separately. This differs
from upstream accuracy-based selection and is fixed before the run. Test data
are evaluated only after checkpoint selection. No architecture search or
additional automatic training runs are configured.

## Validation and results

Eight CPU tests passed on 2026-11-03, including actual pinned-source model forward
and backward on a synthetic `1×3×120×21×2` tensor, source-cache hash/population
checks, exact normalization/masking, storage rejection, explicit zero-frame
retention, internal-missing-frame accounting and zero filling, cache resumption,
deterministic temporal preprocessing, and broadcast gradient parity.
The source-verification regression accepts LF and CRLF checkouts and rejects
semantic edits, other byte changes and a mismatching pinned Git blob.
These checks do not establish recognition performance or real-data throughput.

The completed workflow records cache/sanity provenance, job IDs, histories,
checkpoints, measured learning curves, metrics and per-class outputs under
`/projects/jointposeaction/experiments/phase1c/recognition-baselines/b1/`.
The same run's `comparison/` directory contains the generated reports and paired
uncertainty estimates. Source evidence received locally is the user's supplied
report, accounting and history snapshots; full history files/curve images and a
separate raw-prediction audit are not claimed to have been inspected locally.
See [the matched comparison](DUAL_BACKBONE_COMPARISON.md) and
[rounded measurements](../../reproducibility/experiments/phase1c_completed_243533.json).

## Reproduction

After obtaining the pinned source checkout, set `JPA_HANDFORMER_ROOT` to its root.
The consolidated Phase 1C submission entry point performs source/cache checks
and records the training/evaluation jobs. The cache stage is also callable as:

```bash
python -m phase1c.handformer_data --dataset /path/to/frozen/dataset.json \
  --output "$JPA_ROOT/data/processed/phase1c-handformer" --project-root "$JPA_ROOT"
```

The resulting `compatibility.json` is supplied to the shared Phase 1C
training/evaluation interface. Large caches, checkpoints and predictions remain
outside Git; source, configuration and upstream fingerprints are committed.
