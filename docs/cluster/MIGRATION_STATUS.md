# Phase 1A migration status — 2 November 2026

**Not operational yet.** This is the first interactive checkpoint: local
readiness and an SSH reachability attempt, before resource/storage decisions.

| Stage | Evidence / state |
|---|---|
| A — Local repository | Baseline `5567cb1d6b6bb2b445a0fd3244dc7ee79c4b4b26` matches GitHub main; preservation checks recorded separately |
| B — Access/resources | Endpoint reachable, banner `login-1`; noninteractive authentication failed; user terminal login needed |
| C — Storage | No current allocations/permissions verified; nothing allocated/configured |
| D — Cluster restore | No cluster clone or dependency restore performed |
| E — Environment/Slurm | No environment installed; no job submitted; no GPU validation logs exist |
| F — Acquisition/pilot | Not started at this decision boundary; historical subset is not an approved benchmark |
| G — Sample validation | No new dataset acquired or validated |
| H — Research interfaces | Pending source/schema and pilot evidence; no model implementation started |

See [LOCAL_READINESS.md](LOCAL_READINESS.md) for source preservation, operating
system issues and checks; [RESOURCE_DISCOVERY.md](RESOURCE_DISCOVERY.md) for SSH
evidence, policy sources and the next user-executed command batch.

Ranked blockers:

1. Authenticate in the user's terminal and obtain actual resource/storage output.
2. Confirm assigned SSD path and quota, then review workspace and install plan.
3. Restore a verified commit and build/test a minimal environment under Slurm.
4. Verify official action splits, pose schema and per-recording calibration;
   review an exact pilot manifest and storage estimate before video downloads.
5. Validate a small synchronized sample before authorizing baseline implementation.

No training, reconstruction/tracking implementation, specialized model install,
large download, unrelated project modification or research-source rewrite was
performed. Subsequent setup/data reports must contain actual evidence when
those stages run; they are not represented as completed by placeholder reports.

The next action is authenticated read-only discovery, not model development.
