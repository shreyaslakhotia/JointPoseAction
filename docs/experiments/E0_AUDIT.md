# E0 baseline audit: 3 November 2026

## Observed run, not yet independently audited

User-supplied cluster accounting reports all three stages `COMPLETED`, `0:0`:

| Stage | Job | Elapsed |
|---|---|---|
| Acquire/preprocess | 242798 | 1h52m36s |
| Sanity + baseline | 242871 | 7m15s |
| Final evaluation | 242872 | 54s |

The exported report records 100 classes, retained train/validation/test counts
22,083 / 6,880 / 9,760, Top-1 28.08%, Top-5 54.89%, macro-F1 0.0922 (9.22%),
training-majority Top-1 8.05%, and uniform chance Top-1 1%. These are rounded
report values, not a local independent recomputation. Some classes with 60-90
test examples have zero F1. That is a material limitation of the baseline.

Training source commit: `b188995a566e74de9d10f40baf0884319f2e576c`.
Dataset SHA-256: `f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`.
The audit does not change those artifacts or submit another trained model.

## What the history establishes

The user supplied all 25 epoch records. Training loss fell from 3.910 to 2.450;
online training accuracy rose from 11.14% to 35.77%. Minimum validation loss was
3.10965 at epoch 15, with validation Top-1 26.09% and online training Top-1 30.29%.
Maximum validation Top-1 was 28.30% at epoch 23. The ten non-improving loss epochs
after epoch 15 explain stopping at epoch 25. The chosen checkpoint obeyed the
declared minimum-validation-cross-entropy rule; it was not selected by accuracy.
Online training accuracy includes dropout and weight updates, so it is not the
same as evaluating the final checkpoint on the training split.

The TCN has 95,268 parameters for 216 inputs and 100 classes. Twenty-five epochs
of 691 batches imply 17,275 baseline updates. A seven-minute GPU allocation is
plausible for this network; elapsed time alone does not indicate omitted training.

## Confirmed design limitations and validation gap

- Unweighted cross-entropy and validation-loss selection optimize frequency-weighted
  error, not macro-F1. A low macro-F1 alongside higher Top-1 is mathematically
  possible. Actual prediction concentration needs the saved probability matrix.
- The 16-example/four-class sanity gate establishes bounded optimizer operation,
  not full-corpus alignment or 100-class learnability.
- 11,787 of 50,510 candidate segments were removed: 23.34% overall. Per-class and
  per-split exclusion effects require examination; global retained support is
  insufficient to rule out selection bias.
- `timestamp_cadence` checks each adjacent step within 2 ms of a nominal 60 FPS
  step, but does not constrain accumulated drift. A synthetic 59 FPS sequence
  passes that gate while drifting by over 0.8 seconds across 3001 frames. This is
  a confirmed weakness in the validator, **not evidence that the real corpus has
  that drift**. Absolute pose/video clock alignment was never independently proven.
- This small TCN and feature normalization do not reproduce HandFormer. The
  [HandFormer paper](https://arxiv.org/html/2403.09805v2#S5.SS1) reports pose-only
  action accuracy around 28% on its own protocol and discusses object ambiguity.
  Its full benchmark, architecture and preprocessing differ, so that number is
  not a like-for-like reference or validation of this E0 result.

Static inspection found no obvious transpose, label-reindex, softmax/argmax or
macro-F1 formula error. This is not a finding that every real export is correct.
Do not select a new checkpoint or tune against test performance during this audit.

## CPU-only audit prepared

`scripts/cluster/e0_audit.sbatch` reads only cached project files, requests four
CPUs with a two-hour limit and writes a separate `experiments/e0/audit-JOBID/`.
It performs no download, optimizer step, GPU execution, model selection or edit
to the frozen baseline artifacts.

The audit independently recomputes metrics from saved probabilities/labels and
cross-checks prediction CSV IDs. It reports balanced accuracy, weighted F1,
unpredicted/zero-TP classes, prediction concentration and frequency/F1 association.
It independently streams the original official CSVs to verify retained source IDs,
action/verb/noun mappings, intervals, split/participant identities and training-only
vocabulary selection; consistency against the processed dataset alone is insufficient.
All feature files are hash/shape/mask/index checked, including duplicate inputs
with conflicting labels and cross-split duplicates. Exclusions are summarized by
split, recording and selected class.

All raw source hashes and timestamp spans are checked; accumulated drift is
measured separately from adjacent cadence. First/last retained segments per
recording are reconstructed with an independent per-frame implementation and
compared to exported arrays. The saved checkpoint is evaluated on the full train
and validation splits, and sampled test probabilities are compared to the saved
export. Validation-only time reversal/shuffle and coordinate-zeroing diagnostics
measure sensitivity, with an explicit out-of-distribution interpretation limit.

The complete audit entrypoint is exercised using a synthetic cached run and an
untrained synthetic checkpoint. No local synthetic number is a real E0 result.
Actual audit findings are pending the cluster audit output. Until then, the
responsible conclusion is: a weak, class-uneven baseline was produced; its root
cause has not yet been established.
