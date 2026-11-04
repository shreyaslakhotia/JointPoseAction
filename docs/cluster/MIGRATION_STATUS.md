# Phase 1A migration status — 2 November 2026

> **Current consolidated status (4 November 2026):** E0 and both Phase 1C
> recognition runs are complete. The [final comparison](../experiments/DUAL_BACKBONE_COMPARISON.md)
> records the measured results and frozen population (22,083 train / 6,880
> validation / 9,760 test segments; 100 classes). The phase log below preserves
> its original observations; statements about pending acquisition, validation
> or training describe earlier stages, not current readiness. Use the
> [environment guide](../cluster/ENVIRONMENT_SETUP.md) for setup and the final
> experiment reports for current outcomes. Absolute video/pose synchronization
> and metric hand/object calibration remain unverified.

Latest E0 checkpoint: CPU/GPU setup and authenticated inspection 242218 passed;
26 tests passed on cluster. Official action CSV coverage and 337 cleaned-pose
members are established. The user selected official recording-disjoint splits
and authorized one sanity test plus one baseline run after validity gates pass.
Finite zeros are preserved; unknown coordinate/validity/time semantics still
block training. A cached-data feasibility analysis is prepared, with 28 updated
local tests passing. See `docs/experiments/E0_SCHEMA_VALIDATION.md`; historical
stage descriptions below are superseded where they differ.

**GPU validation pending; CPU environment setup completed.** User accounting
reports installation job 241924 `COMPLETED`, `0:0`. Repository clone and exact
commit `b5e46b1c5eebd34027ee9e148c767ff70ef0c666` check succeeded on cluster.
GPU smoke job 241925 remains queued at the latest supplied status and is unchanged.
User authorized CPU E0 development while waiting, with no new GPU jobs or training.
The segment pipeline/TCN framework now passes 24 local CPU tests; actual dataset
contracts and coverage remain blocked on authenticated payload/sample inspection.

| Stage | Evidence / state |
|---|---|
| A — Local repository | Baseline `5567cb1d6b6bb2b445a0fd3244dc7ee79c4b4b26` matches GitHub main; preservation checks recorded separately |
| B — Access/resources | User login on `login-1`; `ug-proj` / `ug`, two running jobs, five submitted jobs; permitted GPU models exposed; Conda/GitHub verified |
| C — Storage | Dedicated 50 GB SSD path resolves correctly; bootstrap writability/configuration execution pending |
| D — Cluster restore | User cloned and checked exact E0 setup commit; specialized external model repos remain deferred |
| E — Environment/Slurm | Installation 241924 completed `0:0`; logs/version exports need review; GPU 241925 queued |
| F — Acquisition/pilot | Public revision and exact CSV/archive sizes verified; user approved 183.30 MB metadata; authenticated files/sample pending |
| G — Sample validation | No new dataset acquired or validated |
| H — Research interfaces | Segment loader, preprocessing, leakage gates, TCN train/evaluate/metrics/config/Slurm framework implemented; 24 CPU tests passed, no training |

See [LOCAL_READINESS.md](LOCAL_READINESS.md) for source preservation, operating
system issues and checks; [RESOURCE_DISCOVERY.md](RESOURCE_DISCOVERY.md) for SSH
evidence, policy sources and the next user-executed command batch.

Ranked blockers:

1. Authenticate the HF client privately on project storage and run CPU metadata/sample inspection.
2. Inspect actual cleaned schema, units, validity and nominal 30/60 FPS mapping; quantify real split coverage and class distributions.
3. Select and review required pose-member transfer sizes; construct a validated, leakage-controlled source manifest.
4. Retain the queued GPU check; review its result and environment logs without new GPU submissions.
5. Preprocess real data and later authorize the tiny learning sanity test and full E0 run. No training is authorized in this waiting-stage task.

No training, new GPU submission, reconstruction/tracking implementation, specialized
model install, pose archive/video download, unrelated project modification or
historical research-source rewrite was performed. See `docs/experiments/` for the
actual CPU implementation/test status and explicit absence of measured E0 results.

The next command batch is in `PHASE1B_EXECUTION.md`. Only the dedicated
JointPoseAction project is authorized; independent projects must remain untouched.
The other reports distinguish prepared instructions from executed results.
