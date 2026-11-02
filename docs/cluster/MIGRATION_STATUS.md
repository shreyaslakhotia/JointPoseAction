# Phase 1A migration status — 2 November 2026

**Not operational yet.** Phase 1B now authorizes the hand-only E0 baseline after
setup and dataset validation. Password login, account/QOS, dedicated 50 GB SSD
canonical path, live nodes, Conda and GitHub access are confirmed. CPU installation
and dependent L40 smoke-job scripts are prepared; execution evidence is pending.

| Stage | Evidence / state |
|---|---|
| A — Local repository | Baseline `5567cb1d6b6bb2b445a0fd3244dc7ee79c4b4b26` matches GitHub main; preservation checks recorded separately |
| B — Access/resources | User login on `login-1`; `ug-proj` / `ug`, two running jobs, five submitted jobs; permitted GPU models exposed; Conda/GitHub verified |
| C — Storage | Dedicated 50 GB SSD path resolves correctly; bootstrap writability/configuration execution pending |
| D — Cluster restore | No cluster clone or dependency restore performed |
| E — Environment/Slurm | Workspace/cache configuration and GPU smoke-test source prepared locally; no environment installed or job submitted |
| F — Acquisition/pilot | Official source/schema reports prepared; dataset is gated; exact sessions/sizes and access remain pending |
| G — Sample validation | No new dataset acquired or validated |
| H — Research interfaces | Minimal contracts documented; source-specific mappings remain pending; no model implementation started |

See [LOCAL_READINESS.md](LOCAL_READINESS.md) for source preservation, operating
system issues and checks; [RESOURCE_DISCOVERY.md](RESOURCE_DISCOVERY.md) for SSH
evidence, policy sources and the next user-executed command batch.

Ranked blockers:

1. Run the clone/setup submission batch in the user's authenticated terminal.
2. Inspect CPU install and GPU smoke-job results; diagnose any failure before retrying.
3. Confirm authenticated dataset metadata access on the cluster; acquire only bounded metadata first.
4. Verify official action splits, pose schema and per-recording calibration;
   review an exact pilot manifest and storage estimate before video downloads.
5. Validate a small synchronized sample before authorizing baseline implementation.

No training, reconstruction/tracking implementation, specialized model install,
large download, unrelated project modification or research-source rewrite was
performed. Subsequent setup/data reports must contain actual evidence when
those stages run; they are not represented as completed by placeholder reports.

The next command batch is in `PHASE1B_EXECUTION.md`. Only the dedicated
JointPoseAction project is authorized; independent projects must remain untouched.
The other reports distinguish prepared instructions from executed results.
