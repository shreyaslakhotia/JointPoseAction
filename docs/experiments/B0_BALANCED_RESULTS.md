# B0-balanced: controlled TCN class-imbalance experiment

Status: training job 243531 and evaluation/comparison job 243533 completed
successfully. The user supplied the generated cluster comparison on 4 November
2026. The audited original E0 checkpoint and dataset remain unchanged.

## Measured outcome

All models are evaluated on the same 9,760 action segments from 67 recordings.
Values below are rounded cluster-report measurements, not a new local evaluation.

| Model | Top-1 | Top-5 | Macro-F1 | Weighted-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| B0-original | 28.08% | 54.89% | 9.22% | 22.47% | 10.44% |
| B0-balanced | 21.53% | 49.26% | 11.44% | 21.42% | 12.41% |

The balanced recipe improves macro-F1 by 2.22 percentage points and average
class recall by 1.97 points, but reduces Top-1 by 6.56 points. The paired
recording-cluster 95% interval for its Top-1 difference is [-7.57, -5.53] points.
This is an unfavorable overall-accuracy tradeoff, not a replacement for the
original baseline. The treatment includes both weighting and selection changes.

Classes with at least one correct prediction rise from 55 to 92 out of 100;
never-predicted classes fall from 19 to zero. Top-ten prediction concentration
falls from 72.87% to 41.42%. Predicting all classes does not imply strong recall
or precision for all classes, as the 11.44% macro-F1 makes clear. Eight classes
still have zero correct predictions. Verb/noun accuracies are 45.74%/37.36%.

The validation macro-F1 selected epoch 54. Parameters: 95,268. Reported fitting
time: 907.11 seconds; total GPU job elapsed: 17m19s including sanity/startup.
GPU: NVIDIA L40S; peak PyTorch allocated CUDA memory: 85,140,480 bytes.
The supplied first/selected/last history snapshots confirm all 60 epochs ran.
The sanity gate passed after 20 updates (0.77 seconds). At selected epoch 54,
online training Top-1/macro-F1 are 34.37%/27.55%, versus validation
20.54%/10.28%. At epoch 60 these are 34.81%/28.07% and 20.54%/10.14%.
Thus further training improvement did not translate to further validation F1.
The train/validation gap is qualitative: online training uses dropout and weight
updates and is not a post-fit evaluation of the checkpoint on the training set.
The small decline near the end does not justify extending this controlled run.
Full histories and curve images remain on the cluster; only supplied snapshots
were reviewed locally.

Selected checkpoint SHA-256:
`ebb3fb23d226ec947df67167a5a2073fabe3021e96f4a0e961109ae691d9cc6c`.

The original 100-class recording-disjoint dataset has SHA-256
`f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`.
There are 22,083 training, 6,880 validation, and 9,760 test segments. Participants
overlap across splits. Absolute pose/video clock alignment remains unverified;
features are dimensionless, with no claim of metric geometric calibration.

## Prespecified controlled change

`configs/b0_balanced.json` retains the original 64-step, 216-feature tensor,
TCN width 64, dilations 1/2/4, dropout 0.2, batch size 32, AdamW learning rate
0.001, weight decay 0.0001, cosine schedule, seed 42, maximum 60 epochs,
patience 10, and gradient clipping at norm 1.0.

The optimization loss uses class weights `w[c] = n[c]**(-1/2)`, divided by
their arithmetic mean. All `n[c]` are counted exclusively from training rows;
the exact frequencies and weights are captured in training provenance. This
smooths the inverse-frequency correction, reducing the risk of extreme weights
on the least frequent classes. Each batch uses the PyTorch weighted-mean
denominator, the sum of weights for that batch's target labels. Weight
normalization does not alter that weighted-mean objective.

Checkpoint selection changes to **maximum validation macro-F1 over all 100
selected classes**. A strict improvement resets patience; a tie retains the
earlier checkpoint. Validation Top-1 and ordinary cross-entropy remain visible.
This is a controlled treatment of both the optimization objective and selection
criterion, as requested. Any difference cannot be attributed to weighting alone.
No search over weighting strengths or checkpoint-selection criteria is made.

## Sanity and evaluation protocol

A bounded overfit gate uses at most 16 distinct training feature tensors from
four classes, temporary dropout disabled, and at most 500 AdamW updates. It must
reach at least 95% accuracy and reduce loss to at most 25% of its initial value.
The baseline starts with a fresh seed, model and optimizer after the gate; no
sanity weights are reused. The gate does not establish learnability of the full
100-class problem.

`phase1c.train` records plain, unsmoothed, unweighted cross-entropy separately
from the actual optimization loss. Training metrics are online, with dropout
and parameter updates; they are not post-fit training inference metrics.
`phase1c.evaluate` only evaluates the validation-selected completed checkpoint.
It rejects mismatched dataset hashes, vocabularies and compatibility manifests,
and refuses to overwrite outputs. Test predictions are never used for epoch
selection or to launch an automatic replacement run.

## Results and reproducibility artifacts

The existing original TCN result is Top-1 28.08%, Top-5 54.89%, macro-F1 9.22%,
balanced accuracy 10.44%, 19 classes never predicted and 45 classes with no
correct test predictions. Its ten most predicted classes receive 72.87% of
predictions. These are historical measured values, not B0-balanced results.

Measured runtime outputs are `train/provenance.json`
(packages, source revision, GPU, dataset hash, exact configuration, counts and
weights), `train/sanity.json`, `train/history.jsonl`, `train/best.pt`,
`train/training_completed.json`, and `test/{metrics.json,probabilities.npz,
predictions.csv,per_class.csv,evaluation_completed.json}` plus learning curves
and a confusion plot. The final common report compares Top-1/Top-5, macro and
weighted F1, balanced accuracy, unpredicted and zero-correct classes,
prediction concentration, per-class recall/F1 and verb/noun accuracy on the
same ordered held-out segments. The completed run is
`/projects/jointposeaction/experiments/phase1c/recognition-baselines/b0/`.
See [the matched comparison](DUAL_BACKBONE_COMPARISON.md) and the recorded
[rounded measurements](../../reproducibility/experiments/phase1c_completed_243533.json).
