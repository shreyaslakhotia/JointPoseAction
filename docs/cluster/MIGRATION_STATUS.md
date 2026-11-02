# Phase 1A migration status — 2 November 2026

**Not operational yet.** Password login, account/QOS and a dedicated 50 GB SSD
allocation are confirmed by user terminal output. Source/schema research and
local migration tooling have progressed; installation and sample validation
still require actual cluster execution.

| Stage | Evidence / state |
|---|---|
| A — Local repository | Baseline `5567cb1d6b6bb2b445a0fd3244dc7ee79c4b4b26` matches GitHub main; preservation checks recorded separately |
| B — Access/resources | User password login on `login-1`; `ug-proj` / `ug`, two running jobs, five submitted jobs; live nodes pending |
| C — Storage | User created `jointposeaction`, 50 GB SSD, 0 B used; canonical path/access and configuration execution pending |
| D — Cluster restore | No cluster clone or dependency restore performed |
| E — Environment/Slurm | Workspace/cache configuration and GPU smoke-test source prepared locally; no environment installed or job submitted |
| F — Acquisition/pilot | Official source/schema reports prepared; dataset is gated; exact sessions/sizes and access remain pending |
| G — Sample validation | No new dataset acquired or validated |
| H — Research interfaces | Minimal contracts documented; source-specific mappings remain pending; no model implementation started |

See [LOCAL_READINESS.md](LOCAL_READINESS.md) for source preservation, operating
system issues and checks; [RESOURCE_DISCOVERY.md](RESOURCE_DISCOVERY.md) for SSH
evidence, policy sources and the next user-executed command batch.

Ranked blockers:

1. Obtain live node/GRES output and verify dedicated SSD path/access in the user terminal.
2. Review minimal installation plan after node/driver evidence; configure caches on dedicated SSD.
3. Restore a verified commit and build/test a minimal environment under Slurm.
4. Verify official action splits, pose schema and per-recording calibration;
   review an exact pilot manifest and storage estimate before video downloads.
5. Validate a small synchronized sample before authorizing baseline implementation.

No training, reconstruction/tracking implementation, specialized model install,
large download, unrelated project modification or research-source rewrite was
performed. Subsequent setup/data reports must contain actual evidence when
those stages run; they are not represented as completed by placeholder reports.

The next command batch is in `RESOURCE_DISCOVERY.md`. Only the dedicated
JointPoseAction project is authorized; independent projects must remain untouched.
The other reports distinguish prepared instructions from executed results.
