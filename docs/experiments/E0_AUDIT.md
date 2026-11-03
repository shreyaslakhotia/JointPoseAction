# E0 baseline audit: 3 November 2026

## Completed audit: job 243446

The user supplied the successful CPU audit output on 3 November 2026:
`243446 | COMPLETED | 0:0 | 00:07:10`. The complete report remains on the
cluster at `experiments/e0/audit-243446/audit.json`; the supplied compact stdout
is preserved as `reproducibility/experiments/e0_audit_243446.json` with its source
identified. This repository does not claim to have downloaded the full report.

Independent saved-probability recomputation matches Top-1 28.0840%, Top-5
54.8873%, and macro-F1 9.2211%. Prediction CSV mismatches: zero; saved confusion
matrix matches. Weighted F1 is 22.4708% and balanced accuracy is 10.4419%.
Nineteen classes are never predicted and 45 classes have no true positives;
72.8689% of predictions fall in the ten most predicted classes.

Original official CSV hashes, retained intervals/IDs/labels and training-only
vocabulary were verified, with no recording-level split overlap. All 38,723
features passed the reported frame-index/mask checks. Raw source hashes match;
sampled independent feature reconstructions have no failures. There are 264
exact duplicate feature groups, 209 with different labels, and none crossing
splits. Overlapping segments can have indistinguishable pose inputs.

No retained recording exceeds 50 ms accumulated timestamp drift. The overall
maximum 117.05 seconds includes acquired recordings excluded from the retained
population. This is not evidence of such drift in the training set. Absolute
pose/video clock alignment remains independently unverified.

The selected checkpoint achieves train Top-1/macro-F1 32.5409%/13.5244% and
validation 26.0901%/7.6349%. CPU validation Top-1 is reproduced exactly; 300
class-stratified test predictions match saved predictions with maximum
probability discrepancy 1.2e-7. That 300-example sample is an integrity check,
not a replacement test performance estimate. Validation accuracy drops to
16.9767% on time reversal, 18.6773% on time shuffle and 0.8430% with coordinates
removed and masks retained. These are frozen-model, potentially out-of-distribution
sensitivity probes, not trained ablations.

Conclusion: the metrics are credible, while full-vocabulary recognition is weak
and predictions concentrate on frequent classes. Training performance is also
weak; this is not solely a held-out generalization failure. No single causal
explanation was established. The separate Phase 1C brief authorizes two new
controlled recognition runs; this audit itself performed no training.

## Original baseline accounting

User-supplied cluster accounting reports all three stages `COMPLETED`, `0:0`:

| Stage | Job | Elapsed |
|---|---|---|
| Acquire/preprocess | 242798 | 1h52m36s |
| Sanity + baseline | 242871 | 7m15s |
| Final evaluation | 242872 | 54s |

The exported report records 100 classes, retained train/validation/test counts
22,083 / 6,880 / 9,760, Top-1 28.08%, Top-5 54.89%, macro-F1 0.0922 (9.22%),
training-majority Top-1 8.05%, and uniform chance Top-1 1%. These are rounded
report values; the completed cluster audit above independently recomputes them.
Some classes with 60-90
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
  possible. The completed audit confirms that concentration above.
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
Actual cluster audit findings are recorded at the top of this document. The
weak, class-uneven baseline is preserved; its dominant root cause has not been
established by these diagnostics.
