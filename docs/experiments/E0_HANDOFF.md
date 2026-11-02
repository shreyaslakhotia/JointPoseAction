# E0 progress and unattended execution

## Current state

The dedicated 50 GB SSD workspace and Python/CUDA environment are operational;
L40S GPU smoke job 241925 passed. Preparation job **242798** was reported running.
No real E0 training or recognition result has been reported yet.

Official Assembly101 human action annotations supply the targets. Cleaned hand
poses supply the inputs. Candidate coverage after excluding two anomalous training
recordings is 194 / 58 / 83 recordings and 28,194 / 9,060 / 13,256 action segments
across official train / validation / test splits, for 100 training-selected classes.
These counts precede quality exclusions. Acquisition requires 7.994 GB transfer,
23.098 GB extracted, plus processed features and current environment usage.

## Completed work

- Restored the dedicated cluster environment, authentication and Slurm scripts.
- Verified official schemas, canonical-view deduplication, overlapping action
  segments, recording-disjoint splits and reported shared participants.
- Inspected complete pose samples and timestamp/confidence/transform companions.
- Implemented resumable selective acquisition, CRC/SHA checks, confidence masking,
  cadence/boundary exclusions, training-only vocabulary and final support gates.
- Implemented dimensionless 216-feature hand inputs, lightweight residual TCN,
  validation checkpoint selection, held-out metrics/baselines and learning plots.
- Added a gated unattended dependency chain and measured final-result summary.

## Hiccups and resolutions

| Issue | Resolution |
|---|---|
| HF CLI missing click | Pinned CLI dependency repaired by CPU job |
| Slurm selected base Python | Scripts now invoke the absolute project interpreter |
| Sample inspection supplied only train.csv | Corrected shared three-split loader contract; regression added |
| Zero wrists appeared numerically valid | Inspected companions; zero-score hands are conservatively masked, positive-score zeros preserved |
| Two recordings had incompatible timing/coverage | Excluded them rather than recovering/warping their clocks |
| Physical pose units unverified | Hand-length-normalized dimensionless E0 features; no metre claim or metric E3 fusion |
| Usage check measured a symlink | Preparation measures the resolved project and reserves quota headroom |

Local validation: 48 CPU tests passed, including synthetic extraction/preprocessing,
scale invariance, pipeline failure gates, dependency construction, duplicate-run
prevention and final-report assembly. GPU training has not been exercised locally.

## One-command continuation

After pulling the published pipeline commit, run from the cluster repository:

```bash
bash scripts/cluster/submit_e0_pipeline.sh --prepare-job 242798
```

It attaches to the existing preparation job and submits only:

1. One L40 GPU job, after successful preparation: validate the artifact, run a
   training-only overfit gate (up to 16 distinct examples, 500 updates), then train
   one fresh baseline. Sanity uses dropout disabled; baseline retains the fixed
   configuration and starts from a fresh seed/weights. No sanity weights transfer.
2. One CPU job, after successful training: evaluate the minimum-validation-loss
   checkpoint on test once and write metrics, plots and a descriptive error report.

The baseline config remains seed 42, 60 maximum epochs, patience 10, batch 32,
AdamW learning rate .001, width 64 and dilations 1/2/4. The GPU allocation has a
six-hour limit; final CPU evaluation has a one-hour limit. Queue delays remain
possible. Jobs exit as soon as work completes; these limits are not forecasts.

Failure stops the dependent chain: no automatic class/architecture changes,
hyperparameter search, repeated GPU training or E3 work. The final result is not
guaranteed if a gate fails. Preparation logs/exclusions remain available for review.
Repeated submission reads the saved job IDs instead of starting a second baseline;
partial submission can be resumed using the same command and pinned commit.

Slurm `afterok` dependencies and invalid-dependency cancellation follow the
[official sbatch documentation](https://slurm.schedmd.com/sbatch.html#OPT_dependency).
Both dependent IDs are recorded before their temporary submission holds are released.
If preparation has already completed and aged out of Slurm's dependency cache,
the durable ready report is checked before submission and again before execution.
Keep the repository at the pinned commit until all stages finish.

## Outputs on return

All pipeline outputs are outside the repository:
`/projects/jointposeaction/experiments/e0/controlled-baseline/`.

- `submission.json`: preparation/GPU/evaluation job IDs and pinned commit.
- `status.json`: last entered pipeline stage; inspect Slurm accounting as well,
  because preparation failure/cancellation can prevent dependent stages entering.
- `sanity/sanity.json`: actual overfit outcome and training-only segment IDs.
- `train/`: validation-selected checkpoint, config/provenance, history and metrics.
- `test/`: held-out Top-1/Top-5, macro-F1, baseline comparisons, predictions and plots.
- `RESULTS.md`, `results_summary.json`: final measured summary and common confusions.

Performance is on the curated pose-only subset, with official recording split
membership and shared participants. Participant independence, metric calibration
and independently verified RGB clocks are not claimed. Temporal indexing follows
the declared reference-reader protocol with actual cadence and interval checks.
