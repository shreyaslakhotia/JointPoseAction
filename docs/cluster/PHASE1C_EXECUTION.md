# Phase 1C: matched TCN and HandFormer baselines

The production workflow completed successfully at source commit
`fcb0a40db35c1c20da76df5fa90fa7a9179d63ad`: prepare 243530, balanced TCN 243531,
HandFormer 243532 and evaluation/comparison 243533. Measured outcomes are in the
[matched results report](../experiments/DUAL_BACKBONE_COMPARISON.md).

The instructions below document the original submission and guarded recovery.
The existing submission is complete and pins its recorded source commit; it
should not be rerun to create replacement scientific experiments. Reproduction
uses the recorded source/configuration and an explicitly separate run plan.
The user's authenticated cluster shell performed the submissions; no local
synthetic result is presented as a real recognition result.

```bash
(
  set -e
  cd /projects/jointposeaction/repo
  git pull --ff-only
  bash scripts/cluster/submit_phase1c.sh
)
```

The dedicated SSD, existing environment and frozen original E0 paths are used.
The original E0 run and its source modules are not overwritten. All Phase 1C
outputs live under:

`/projects/jointposeaction/experiments/phase1c/recognition-baselines/`

## Recorded scientific recipes

- B0-original: frozen epoch-15 TCN, 95,268 parameters, minimum validation CE.
- B0-balanced: same architecture, 64x216 features, dataset and major training
  settings; inverse-square-root training-frequency weights normalized to mean
  one; maximum validation macro-F1 selects the checkpoint, patience ten, at most
  60 epochs. This compares weighting **and** selection changes together.
- B1-HandFormer: actual official pose-only `HF_Pose` at pinned upstream
  `640a10a6bd03f85842a2662e67cf30444ab1d673`, 3,805,866 parameters and
  3x120x21x2 joint inputs. Published SGD, five-epoch warmup, epoch-index milestones
  25/40, smoothing/temperature loss, 50 epochs, and micro-action settings are
  retained. Validation macro-F1 selects a checkpoint. There is no class weighting.

Both new models use seed 42 and exactly the frozen 100-class example population:
train 22,083, validation 6,880, test 9,760. SHA-256:
`f45c749371568c29f840d2613d9ed20e9859e9020d1555f039882196cd133b59`.
No label selection or class weighting uses validation/test frequencies.

HandFormer uses the same source action intervals, but takes full source sequences
before its authored temporal crops and interpolation. It uses E0's shared origin,
dimensionless scale and world axes instead of the authors' smoothed left-wrist
transforms. No extra temporal context is added. These are explicit preprocessing
differences; a model-family comparison does not isolate architecture alone.
See `docs/experiments/B1_HANDFORMER_RESULTS.md` and the upstream provenance JSON.

## Four jobs, one submission

| Stage | Resources | Dependencies | Time ceiling |
|---|---|---|---|
| `prepare` | 4 CPUs | none | 3 hours |
| `b0` | 1 L40 GPU | successful prepare | 6 hours |
| `b1` | 1 L40 GPU | successful prepare | 24 hours |
| `evaluate` | 4 CPUs | both model jobs successful | 6 hours |

Ceilings are safety limits, not measured runtime estimates. Jobs exit as soon as
work finishes. HandFormer throughput will be visible from actual epoch logs.
The two training jobs can run concurrently; normal `ug` Slurm limits cap total
running jobs at two. Four new submitted jobs leave space for one existing job
within the five-job limit. If fewer than four slots are free, the helper fails
before submitting anything; rerun the same entry point when slots are free.
There is no automatic QoS escalation or GPU-type switching.

Each training job first performs its own bounded overfit gate (at most 16 training
examples, 500 updates), records `train/sanity.json`, then initializes a fresh
model and optimizer for its one full run. A failed sanity gate stops that model
before full training. Test inference is not used during preparation, sanity,
fitting or checkpoint selection.

All four jobs are submitted held. IDs and dependencies are saved before any job
is released. Re-running submission reuses recorded IDs rather than launching
duplicate experiments; partial submission resumes missing stages. A changed
source commit or dataset is rejected against the existing submission record.
No low performance condition launches another run. Failed training is not silently
restarted; ordinary failure diagnosis must precede any bounded retry.

## CPU preparation and storage

Preparation verifies feature hashes and the fixed population, snapshots every
original E0 run artifact's hash, and fetches only the small pinned HandFormer
**source repository** into `$JPA_ROOT/upstreams/HandFormer`. It downloads no poses,
RGB, pretrained weights or RGB features. Source size is reported in the log.
Existing Python/PyTorch/NumPy are reused; the minimal compatibility adapter avoids
historical environment replacement and profiling-only dependencies.

The new HandFormer cache stores each retained recording once as float32 source
coordinates, validity masks and integer frame IDs. It is capped at **6 GB**, with
at least **2 GB additional headroom** for checkpoints and reports inside the
50 GB allocation. Actual project use and cache bytes are logged and guarded.
No data are removed to make space. A quota problem stops preparation and keeps
GPU jobs behind the failed dependency. Completed cache records can be resumed
after hash verification; no incomplete files are silently accepted as complete.

CPU forward/backward checks exercise both models before the preparation marker
is written. Full-interval internal missing observations are explicitly zero-filled
and counted, consistent with the original confidence/missing-data policy; valid
zero coordinates are preserved. Population changes, unsafe paths or corrupted
sources are hard failures.

## Recovery of preparation job 243518

The first submission recorded prepare/B0/B1/evaluate jobs
243518/243519/243520/243521. Preparation finished caching all 268 retained
recordings (3,699,724,330 bytes), preserving 22,083/6,880/9,760 examples, then
failed on the HandFormer source fingerprint. All dependent jobs were cancelled
with zero elapsed time; neither training run started.

The provenance hashes had mistakenly been generated from Windows CRLF checkout
bytes. The pinned upstream Git objects and the Linux checkout use LF. The fix
records canonical Git-blob SHA-256 values and independently checks both the pinned
Git objects and working source after **only** CRLF-to-LF conversion. It accepts
no whitespace stripping or other edits. Source verification now precedes cache
processing. The dataset, model configurations, upstream revision and architecture
are unchanged.

After pulling the repair commit, one consolidated command resumes the workflow:

```bash
(
  set -e
  cd /projects/jointposeaction/repo
  git pull --ff-only
  bash scripts/cluster/submit_phase1c.sh --recover-preparation
)
```

This explicitly bounded recovery is restricted to the diagnosed failure at the
original source commit. It verifies your four old Slurm records, requires failed
preparation and cancelled/zero-elapsed dependent jobs, and refuses recovery if a
model's training directory or successful preparation marker exists. A complete
cache with matching dataset hash and split counts is required. Old submission
and failure metadata are archived under `recovery-history/preparation-243518.json`.

The replacement submission pins the repaired source and retains the same output
root/cache. The CPU stage revalidates cached hashes and model compatibility;
successful preparation releases the two scientific runs through dependencies.
Repeated invocation reuses the replacement IDs. A second recovery or any attempt
to replace an executed model is refused. There is no additional scientific run
or hyperparameter change in this repair.

## Outputs and monitoring

- `submission.json`: pinned source, dataset protocol, all four job IDs and release
  status. Separate `status-<stage>.json` files record stage completion/failures.
- `prepared.json`: storage use, source path, CPU model compatibility and config
  hashes. `handformer-cache/compatibility.json` captures cache provenance.
- `b0/train/`, `b1/train/`: sanity, configuration/environment provenance, class
  weights/counts, histories, validation metrics and best checkpoints.
- `b0/test/`, `b1/test/`: final probabilities, segment-ID predictions, comprehensive
  metrics, per-class tables and measured curves/confusion plots.
- `comparison/`: `B0_BALANCED_RESULTS.md`, `B1_HANDFORMER_RESULTS.md`,
  `DUAL_BACKBONE_COMPARISON.md`, `FUSION_READINESS.md`, per-class CSV and comparison
  JSON including paired recording-cluster bootstrap intervals.

```bash
sacct -X -j PREPARE_ID,B0_ID,B1_ID,EVALUATE_ID \
  --format=JobID,State,ExitCode,Elapsed,AllocTRES -P
tail -n 5 /projects/jointposeaction/logs/p1c-b1-B1_ID.out
```

The comparison requires the same 9,760 held-out segment IDs and labels across all
three models by default. It checks saved probability/CSV consistency and does not
refit the original model. Recording-cluster bootstrap preserves overlapping
segments within a recording; it does not establish participant independence or
training-seed uncertainty. The already inspected original test results remain
diagnostic context, never a configuration-selection criterion.

Absolute pose/video synchronization and metric hand/object calibration remain
unverified. Geometry fusion is documented only, with no reconstruction, tracking,
RGB inference or fusion implementation in this phase.
