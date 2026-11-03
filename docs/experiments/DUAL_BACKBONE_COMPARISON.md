# Phase 1C comparison protocol and results status

**Status: implementation prepared; the two new cluster training/evaluation runs have not yet produced measured results.** This file specifies the comparison protocol. A completed comparison is generated from the frozen prediction artifacts into the dedicated Phase 1C experiment directory, with the same filename. No synthetic test outcome is reported as a real experiment.

| Model | Test Top-1 | Test Top-5 | Test macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| B0-original, audited full test population | 28.08% | 54.89% | 9.22% | 10.44% |
| B0-balanced | Pending measured evaluation | Pending | Pending | Pending |
| B1-HandFormer pose-only | Pending measured evaluation | Pending | Pending | Pending |

The original values above are previously audited, rounded results on all 9,760 test segments. They are reference values, not a completed matched comparison with the new models. The final report recomputes every model's metrics from its prediction probabilities on exactly the same segment identities.

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

After both evaluation jobs finish, the unified CPU stage invokes:

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

The original result showed frequent-class concentration and weak training performance. Class balancing is one controlled intervention, not an established sole remedy. B1 may improve overall accuracy, class coverage, both or neither. Report the measured tradeoff without extending the experiment into open-ended tuning.
