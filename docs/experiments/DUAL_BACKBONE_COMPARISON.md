# Phase 1C matched recognition results

**Completed:** prepare 243530, B0-balanced 243531, HandFormer 243532 and evaluation/comparison 243533 all finished `COMPLETED`, `0:0`. The user supplied the generated cluster report on 4 November 2026. This repository records those measured, rounded values; it does not claim a separate local recomputation from raw predictions.

| Model | Test Top-1 | Test Top-5 | Macro-F1 | Weighted-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| B0-original | 28.08% | 54.89% | 9.22% | 22.47% | 10.44% |
| B0-balanced | 21.53% | 49.26% | 11.44% | 21.42% | 12.41% |
| B1-HandFormer pose-only | 36.31% | 65.92% | 17.09% | 32.65% | 17.95% |

The population is `full_frozen_test`: exactly 9,760 segments from **67 recordings**, with all 100 classes. The completed cluster comparison recomputed every model's metrics from its saved probabilities on matched segment identities. The source manifest hash and ordered vocabulary match the frozen E0 protocol.

| Model | Never predicted | Zero-correct classes | Classes correct at least once | Top-ten prediction concentration | Verb Top-1 | Noun Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| B0-original | 19 | 45 | 55 | 72.87% | 50.14% | 43.23% |
| B0-balanced | 0 | 8 | 92 | 41.42% | 45.74% | 37.36% |
| B1-HandFormer | 5 | 14 | 86 | 59.52% | 61.08% | 48.52% |

## Interpretation of the measured tradeoff

HandFormer is the strongest overall recognizer among these fixed runs: Top-1 improves by **8.23 percentage points** and macro-F1 by **7.87 points** over the original TCN. It also improves Top-5, weighted-F1, balanced accuracy and verb/noun recognition. Its 17.09% macro-F1 and 14 zero-correct classes still indicate limited full-vocabulary recognition; improvement is not evidence that the task is solved.

B0-balanced improves macro-F1 by 2.22 points and broadens nonzero class coverage from 55 to 92 classes, but Top-1 falls by **6.56 points**. Predicting every class and getting one example right are weak coverage thresholds, not evidence of reliable recognition for every class. Keep this as an explicit accuracy/coverage tradeoff, rather than silently replacing the original TCN. It does not meet a goal of broader coverage with little overall-accuracy loss.

The balanced intervention combines weighting and macro-F1 checkpoint selection. HandFormer also differs in temporal preprocessing, representation, loss and capacity. These results establish useful reference runs, not isolated causal estimates of weighting or architecture. No additional training or hyperparameter search is triggered by the observed tradeoff.

## Paired accuracy differences

| Difference | Top-1 difference | 95% recording-cluster interval |
|---|---:|---:|
| B0-balanced minus original | -6.56 pp | [-7.57, -5.53] pp |
| HandFormer minus original | +8.23 pp | [+6.94, +9.58] pp |
| HandFormer minus balanced | +14.78 pp | [+13.51, +16.04] pp |

All intervals use 10,000 recording-cluster bootstrap replicates, seed 20261103. They exclude zero for the corresponding accuracy differences, conditional on these fixed models and held-out recording population. They do not establish stability across training seeds or participant-independent generalization. Deltas use the cluster report's underlying values and need not equal subtraction of the rounded table entries.

## Checkpoint and resource record

| Model | Parameters | Selected epoch / criterion | Fitting seconds | GPU job elapsed | Peak allocated CUDA bytes |
|---|---:|---|---:|---|---:|
| Original TCN | 95,268 | 15 / minimum validation CE | Not recorded | 7m15s, job 242871 | Not recorded |
| Balanced TCN | 95,268 | 54 / maximum validation macro-F1 | 907.11 | 17m19s, job 243531 | 85,140,480 |
| HandFormer | 3,805,866 | 41 / maximum validation macro-F1 | 14,696.75 | 4h13m34s, job 243532 | 8,716,908,032 |

All used NVIDIA L40S. Fitting time excludes preparation/startup outside the fitting timer; peak allocated memory is a PyTorch allocator measure rather than total device usage. Evaluation/comparison job 243533 took 20m26s. The original TCN retains substantial compute efficiency, while HandFormer offers the better measured recognition performance.

## Learning behavior and sanity gates

Both sanity gates passed: B0 in 20 updates/0.77 seconds, B1 in 50 updates/7.87 seconds. All configured epochs ran: 60 for B0-balanced and 50 for HandFormer. The selected epochs and validation macro-F1 in the completion records agree with the supplied history snapshots.

| Model / epoch | Online train Top-1 | Online train macro-F1 | Validation Top-1 | Validation macro-F1 | Validation CE |
|---|---:|---:|---:|---:|---:|
| Balanced TCN / selected 54 | 34.37% | 27.55% | 20.54% | 10.28% | 3.497 |
| Balanced TCN / final 60 | 34.81% | 28.07% | 20.54% | 10.14% | 3.491 |
| HandFormer / selected 41 | 49.65% | 29.67% | 35.36% | 16.01% | 2.731 |
| HandFormer / final 50 | 51.87% | 31.93% | 34.45% | 15.42% | 2.829 |

Both models improve on training data after their selected epoch without improving validation F1. The HandFormer pattern, including rising validation CE, is consistent with late overfitting. Validation and test performance are broadly aligned for the selected checkpoints. These sparse snapshots support retaining the declared validation-selected checkpoints, not extending training or claiming a full learning-curve audit. Training metrics are online with dropout, parameter updates and, for HandFormer, augmentation; direct train/validation gaps are not controlled post-fit measurements.

The completion record's `test_evaluated: false` describes the training stage. Test evaluation occurred later in separate job 243533, as intended; this field is not a failure of the final evaluation.

## Error observations

On the same held-out population, `inspect partial toy -> rotate partial toy` falls from 161 original-TCN errors to 73 HandFormer errors. `pick up screw -> put down screw` falls from 91 to 65 with HandFormer, but rises to 123 with the balanced TCN. HandFormer still confuses `rotate partial toy -> inspect partial toy` (66) and `rotate partial toy -> pick up partial toy` (65), and makes 50 `pick up screw -> pick up screwdriver` errors.

These are descriptive counts, not proof that object ambiguity, temporal direction or another specific factor caused the errors. They motivate controlled future geometric comparisons, without establishing that geometry will fix them. Full per-class tables and confusion matrices remain in the cluster artifacts.

## Frozen population and scientific controls

The approved E0 manifest hash is `f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`. Its 100 classes and mappings remain fixed, with 22,083 training, 6,880 validation and 9,760 test action segments. Official recording-disjoint split membership is retained. Some participants occur across splits; participant-independent generalization is not claimed.

`phase1c.compare` checks the manifest hash, evaluation provenance, class ordering when exported, prediction segment IDs, action IDs, recording IDs, integer labels, probabilities and CSV prediction consistency. Row order may differ between models because matching uses segment identity. The source manifest defines the probability-column vocabulary and all true labels. New evaluations export that class mapping explicitly; the original E0 export obtains its ordering from its matching frozen-manifest provenance.

The primary comparison requires every model to predict all frozen test segments. A differing population fails by default. A separately declared `--allow-common-subset --exclusion-reason "..."` produces an explicitly labelled common-subset comparison, preserving excluded IDs per model and recomputing B0-original on that subset. This option documents an already approved deterministic data policy; it does not authorize changing the evaluation population to avoid an implementation problem. The original full-test report remains unchanged. Never compare a full-test score in one row to a common-subset score in another row.

Training classes and balancing weights use training data only. Checkpoints use their documented validation criterion. Neither this report nor the test predictions select checkpoints, class weights, preprocessing or configurations. The original test results have already been inspected, so subsequent comparisons are diagnostic evidence rather than a fresh untouched confirmatory test. No automatic retraining follows an unfavorable result.

## Metrics and recording-cluster uncertainty

The report includes Top-1/Top-5, macro-F1 across all 100 selected classes, support-weighted F1, balanced accuracy over supported classes, never-predicted and zero-correct class counts, top-ten prediction concentration, probability-marginal verb/noun accuracy, per-class recall/F1, and the most frequent off-diagonal confusions. Balanced accuracy and macro-F1 use distinct denominators and must not be interchanged. With the full test population all selected classes have support; any common subset explicitly reports support per class.

For every model pair, the accuracy difference is computed on paired predictions. A fixed-seed (`20261103`) percentile bootstrap uses 10,000 replicates. Each replicate draws recordings with replacement and retains every action segment from each selected recording, including overlapping segments. It recomputes the segment-weighted accuracy difference: total additional correct segments divided by total sampled segments. It does not average recording accuracies, and it does not sample individual action segments independently.

The resulting 95% interval expresses variation across recording clusters conditional on these fixed trained checkpoints and this selected population. Shared participants may preserve dependence between recording clusters. It does not quantify training-seed variability, annotation uncertainty, synchronization error or the effect of trying multiple comparisons. Interval exclusion of zero is not a claim of universal architectural superiority.

## Artifacts and execution

The completed unified CPU evaluation stage invoked:

```bash
python -m phase1c.compare \
  --dataset /path/to/frozen/dataset.json \
  --original /projects/jointposeaction/experiments/e0/controlled-baseline \
  --balanced /path/to/phase1c/b0-balanced \
  --handformer /path/to/phase1c/b1-handformer \
  --output /path/to/phase1c/comparison
```

The output must be outside all input run directories and must not exist already. The comparison creates `comparison.json`, `per_class.csv`, the measured `DUAL_BACKBONE_COMPARISON.md`, model-specific `B0_BALANCED_RESULTS.md` and `B1_HANDFORMER_RESULTS.md`, and a copy of `FUSION_READINESS.md`. It reads each run's `test/{metrics.json,predictions.csv,probabilities.npz}` plus available training provenance, completion record and learning history. Input prediction/checkpoint/provenance hashes are recorded and checked again after reporting. The frozen original E0 artifacts are never rewritten.

The measured report points to each run's training/validation history and plots, records best epoch, selection criterion, parameters, training duration, GPU and peak CUDA memory when available, and exposes missing historical metadata as `not recorded`. It does not manufacture resource measurements. Separate B0 and B1 reports document model-specific preprocessing, loss treatment and upstream revision. Differences in HandFormer temporal sampling or normalization mean that an accuracy difference cannot be attributed to architecture alone.

## Remaining limitations

The audited pose/annotation contract uses nominal reference-reader timing. Absolute pose-to-video synchronization has not been independently verified, and metric hand/object calibration is not established. Positive source confidence and finite coordinates define usable observations; valid zero coordinates are preserved. These limitations carry forward into both baselines and the future fusion study.

The original result showed frequent-class concentration and weak training performance. The completed interventions demonstrate broader coverage with a TCN accuracy penalty, and stronger overall recognition with HandFormer. Dominant causal explanations remain unresolved. Keep all three frozen runs for later within-backbone geometric comparisons, with the existing protocols and selection criteria made explicit.

Measured source record: [phase1c_completed_243533.json](../../reproducibility/experiments/phase1c_completed_243533.json). Large run artifacts remain under `/projects/jointposeaction/experiments/phase1c/recognition-baselines/`; the original run remains under `/projects/jointposeaction/experiments/e0/controlled-baseline/`.
