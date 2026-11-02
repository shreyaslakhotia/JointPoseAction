# Phase 1B execution checkpoint — setup before E0

Latest checkpoint: the user completed clone/setup submission; installation job
241924 finished `COMPLETED / 0:0`, GPU job 241925 remains queued. Do not rerun
the historical setup batch below. CPU E0 pipeline/model/tests are implemented
under `e0/` and `tests/`; the immediate next task is authenticated CPU metadata
inspection using `e0_inspect.sbatch`, not a new GPU submission or training.
The user approved exactly 183,299,707 bytes of split/mapping metadata. Full pose
archive/video downloads remain unauthorized. See `docs/experiments/E0_DATASET.md`.

Baseline migration revision `dc9d95c992a9a380bd3c67fb47c409a9d5223a0d` was checked
against GitHub. Work continues on dedicated `development/e0-baseline`. The user
has authorized minimal cluster setup and a real E0 training/evaluation pipeline.
No E1–E5, reconstruction, tracking or mistake detection is in scope.

Completed checks are reused: SSD canonical path, module availability/Conda version,
Git/GitHub access, association/QOS, live models and preservation validation.
No further audit or GPU benchmarking is needed. Password SSH still requires the
user's terminal; preparing local scripts does not give the agent that session.

## One user-executed batch — cluster login shell

After reviewing the pinned installation plan in `ENVIRONMENT_SETUP.md`, clone the
published E0 branch into the unused dedicated path, then submit CPU setup and its
dependent L40 smoke test. This is an installation/submission batch, not read-only.
Use the exact reviewed source SHA supplied in the handoff and compare after clone.

```bash
(
  set -e
  umask 077
  : "${JPA_EXPECTED_COMMIT:?Set this to the reviewed E0 commit supplied in the handoff}"
  test ! -e /projects/jointposeaction/repo
  git clone --branch development/e0-baseline --single-branch https://github.com/shreyaslakhotia/JointPoseAction.git /projects/jointposeaction/repo
  cd /projects/jointposeaction/repo
  git rev-parse HEAD
  test "$(git rev-parse HEAD)" = "$JPA_EXPECTED_COMMIT"
  bash scripts/cluster/submit_setup.sh
)
```

The subshell stops on failures without changing the user's interactive shell's
error mode. If `repo/` already exists, stop and return `git -C ... status --short`
and its commit instead of cloning a second copy or deleting it. Do not install on
the login node. All caches and installation work are handled by the submitted jobs.

Return submission output/job IDs. Once finished, collect both `.out` and `.err`
logs plus `sacct -X -j INSTALL_ID,SMOKE_ID --format=JobID,State,ExitCode,Elapsed -P`.
Do not paste credentials. If a submission is rejected, return the exact rejection;
do not resubmit with alternate resource/QOS flags.

## After setup passes

Source project cache configuration before privately authenticating with the HF
CLI; access conditions are approved, but cluster file access is not tested yet.
Then list pinned annotation metadata with the inventory script in a CPU job.
Acquire action/class/split metadata and inspect selective cleaned pose access.
E0 needs no raw RGB download for training; acquire video only for a specific
schema/timing validation need. Present measured bytes before substantial downloads.

Resolve pose schema/units/identity and 30 FPS action-to-pose timing, deduplicate
view-level action rows and establish recording/participant separation. Class
selection uses training counts only; count all exclusions. Only then implement
raw-pose preprocessing and a lightweight TCN, test a tiny overfit run, train once
under Slurm and evaluate the validation-selected checkpoint on held-out data.
Review official MS-G3D, HandFormer pose-only and TCN references at that decision
point; no broad model exploration is needed.

E0 is not complete until real training and held-out Top-1/Top-5 evaluation finish.
No E0 result report will present unmeasured metrics as outputs. Training, test
predictions and checkpoints remain outside Git; compact summaries retain commit,
data manifest, selected vocabulary, seed, configs, environment and job IDs.
