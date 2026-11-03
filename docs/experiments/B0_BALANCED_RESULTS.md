# B0-balanced: controlled TCN class-imbalance experiment

Status: implementation prepared; no new measured performance is claimed here.
The audited original E0 checkpoint and dataset are preserved unchanged. The
orchestrator writes the measured result and unified comparison after the jobs
complete; pending fields must not be interpreted as experimental results.

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

New measurements are pending. Runtime outputs are `train/provenance.json`
(packages, source revision, GPU, dataset hash, exact configuration, counts and
weights), `train/sanity.json`, `train/history.jsonl`, `train/best.pt`,
`train/training_completed.json`, and `test/{metrics.json,probabilities.npz,
predictions.csv,per_class.csv,evaluation_completed.json}` plus learning curves
and a confusion plot. The final common report compares Top-1/Top-5, macro and
weighted F1, balanced accuracy, unpredicted and zero-correct classes,
prediction concentration, per-class recall/F1 and verb/noun accuracy on the
same ordered held-out segments. Jobs and Slurm accounting belong in the
submission manifest and runtime report; none have been invented here.
