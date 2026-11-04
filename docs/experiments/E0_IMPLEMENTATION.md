# E0 implementation — CPU tested, no training started

> **Current consolidated status (4 November 2026):** E0 and both Phase 1C
> recognition runs are complete. The [final comparison](../experiments/DUAL_BACKBONE_COMPARISON.md)
> records the measured results and frozen population (22,083 train / 6,880
> validation / 9,760 test segments; 100 classes). The phase log below preserves
> its original observations; statements about pending acquisition, validation
> or training describe earlier stages, not current readiness. Use the
> [environment guide](../cluster/ENVIRONMENT_SETUP.md) for setup and the final
> experiment reports for current outcomes. Absolute video/pose synchronization
> and metric hand/object calibration remain unverified.

Authorization update: the user permits a tiny sanity/overfit run and one
single-GPU E0 baseline once data-validity, leakage, coverage and storage gates
pass. No new permission is needed then. Current work stops before training
because the cleaned coordinate/validity and endpoint/origin contract remains
unverified. Official recording-disjoint splits are retained. See
`E0_SCHEMA_VALIDATION.md` for the current evidence checkpoint.

## Architecture choice

Reviewed the official [MS-G3D baseline](https://github.com/assembly-101/assembly101-action-recognition/tree/00d3c3bb3af75ce38ae406614a7cb034aecdebc0/MS-G3D-action-recognition)
and [HandFormer preprocessing](https://github.com/s-shamil/HandFormer/tree/640a10a6bd03f85842a2662e67cf30444ab1d673/Preprocess).
MS-G3D adds graph machinery; HandFormer adds attention and optional RGB. Its
reference normalization also smooths wrist transforms, so it is not copied for
the raw-pose E0 experiment. The user's preferred lightweight TCN is the smaller
maintenance/dependency choice. This is not a reproduction or competitiveness claim.
The [TCN study](https://arxiv.org/abs/1803.01271) supports temporal convolutions as
a sequence-modeling baseline; this implementation is independently written.

`e0/model.py`: input `[batch,time,features]`, 1x1 projection to width 64, three
residual blocks with dilations 1/2/4 and two kernel-3 convolutions per block,
GroupNorm/ReLU/dropout 0.2, temporal mean pooling and a class-logit head. The
64-sample, 216-feature default has 90,068 parameters for 20 classes. Convolutions
are bidirectional within the annotated clip; this is segment recognition, not
online anticipation. Cross-entropy receives logits, evaluation uses softmax.
Validity channels make missingness explicit; entirely unusable clips fail the
preprocessing threshold rather than becoming valid all-zero examples.

The feature dimension is an argument rather than hardcoded to a hand-only head.
No object feature extraction/fusion, RGB encoder, tracking or reconstruction is
implemented. Future object features can use the same temporal input contract.

## Training and evaluation interfaces

`configs/e0_tcn.json`: seed 42, batch 32, AdamW learning rate 0.001, weight decay
0.0001, cosine schedule, at most 60 epochs, patience 10. No class weighting by
default; optional inverse-frequency weights use training examples only. Seeds
and deterministic algorithms are enabled; unsupported nondeterministic operations
fail instead of being hidden. DataLoader workers are zero for reproducibility.
GPU jobs set the required deterministic cuBLAS workspace configuration.

`python -m e0.train` requires a Slurm GPU context, validated/hash-checked real
dataset, nonempty splits and clean committed source. It creates a fresh run
directory, writes versions/commit/config/vocabulary/data hash/job ID/GPU provenance,
logs train/validation loss and accuracy each epoch, and saves the checkpoint
with minimum validation cross-entropy. No test loader or test checkpoint selection
is used in training. These interfaces are prepared only; no optimizer step has
been executed in this phase.

`python -m e0.evaluate` uses that checkpoint on the held-out test split, requires
the identical prepared dataset/vocabulary hash, and runs on CPU. Outputs include
Top-1, Top-5 with effective k qualification, macro F1 over all selected and
supported classes, per-class F1/accuracy/support, confusion counts, and verb/noun
Top-1 from probability marginals over the official class mapping. Majority class
is selected from training labels; uniform and training-frequency chance baselines
are reported. Unsupported test classes have support zero and undefined accuracy,
not silently omitted from the selected-class macro score.

Artifacts: best checkpoint, JSONL history, validation metrics, test probabilities,
prediction CSV with segment identity and missingness, metrics JSON and static
loss/accuracy/confusion/class-count figures. Large arrays/checkpoints belong in
project storage outside Git. No old experiment output is overwritten. Full
optimizer-state resume is not implemented; a partial run must not be reported as
completed. Training wall-time estimates await actual subset size.

## Prepared Slurm entry points — do not submit training yet

- `e0_inspect.sbatch`: two CPU cores; unit tests plus approved metadata and bounded
  archive inspection, 20-minute limit. No GPU request.
- `e0_preprocess.sbatch`: two CPU cores, one hour; explicit reviewed manifest.
- `e0_train.sbatch`: one L40, two hours; no prohibited GPU CPU/memory flags.
- `e0_evaluate.sbatch`: two CPU cores, 30 minutes; explicit dataset/checkpoint.

No additional GPU jobs were submitted. Job 241925 is unchanged. CPU inspection is
the only immediate next submission proposed, in the user's authenticated terminal.
Dataset/schema validation and a later explicitly authorized tiny overfit/sanity
run are still required before real baseline training.

## CPU validation

Run locally or in a CPU job from repository root:

```bash
python -B -m unittest discover -s tests -v
```

24 tests passed locally using existing Python 3.12 / PyTorch 2.15.0+cpu and NumPy
2.5.4; no new local environment was installed. These differ from cluster pins, so
the CPU inspection job reruns the suite in the installed cluster environment.
Tests cover overlapping/equal-interval examples, view selection, 30/60 FPS mapping,
schema rejection, missing/NaN/zero/wrist validity, translation invariance, leakage,
training-only vocabulary, end-to-end synthetic preprocessing/loader hashes,
failed readiness gates, TCN forward/backward shapes, deterministic seeds,
inference without parameter updates, checkpoint serialization, exact metrics,
verb marginals, baselines, and bounded range/ZIP behavior. No optimizer step,
trained model, held-out dataset evaluation or GPU computation was performed.

All synthetic fixture names/classes/coordinates are marked synthetic. The real
training CLI refuses those artifacts; passing a synthetic test does not validate
physical units, annotation quality, synchronization or real-data learnability.
