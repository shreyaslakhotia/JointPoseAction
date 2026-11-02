# Resource discovery — 2 November 2026

Status: user password login, account/QOS, dedicated SSD canonical path, live GPU
models, Conda and GitHub access confirmed by user terminal output. Installation
and GPU execution remain pending.

## Phase 1B evidence update

- `readlink -f /projects/jointposeaction` returned `/projects/_ssd/jointposeaction`.
  `ls -ld` showed the project link owned by `shreyas010`.
- `conda --version` after loading Miniforge: `conda 24.11.3`.
- `git --version`: `2.43.0`.
- Cluster `git ls-remote` returned migration branch
  `dc9d95c992a9a380bd3c67fb47c409a9d5223a0d`. A fresh local remote query confirmed
  that revision before creating `development/e0-baseline`.
- `sinfo` lists CPU nodes and `l40`, `pro6000`, `rtx5090`, `6000ada`, `a40`,
  `a6000`, `gh200`. L40 and one PRO 6000 node were mixed; other permitted GPU
  nodes were allocated. GH200 was idle but is not established as allowed by the
  user's QOS rows, so it is not selected.
- Plan: CPU install followed by one normally queued L40 smoke test. Snapshot
  occupancy is not a reason to change QOS, reserve idle GPUs or benchmark hardware.
- HF dataset access conditions: user confirmed approval; cluster token/session
  authentication and file access remain untested. No credentials were requested.

No need to repeat these read-only queries. The previous next-verification batch
below is historical and superseded. Follow `PHASE1B_EXECUTION.md` for the next
clone/setup step. Directory writability is checked by the bootstrap itself.

## Current verified evidence (user terminal)

- Authenticated prompt: `shreyas010@login-1`; password-based SSH succeeded.
- `squeue --me` returned no jobs in the supplied earlier snapshot.
- `module avail` lists `Miniforge3/24.11.3-1`, `Miniconda3/25.5.1-0`,
  CUDA through `13.0.0`, including `12.8.0`, plus GCC modules. Module presence
  does not establish a compute node's driver version.
- Association: account `ug-proj`; default QOS `ug`; permitted QOS values
  `ug,override-limits-but-killable`. Other QOS rows in the global query are not
  evidence that this user can select those QOS values.
- `ug`: `MaxJobsPU=2`, `MaxSubmitPU=5`; per-user GPU TRES limits below.
  These are ceilings, not reserved capacity or a promise a submission will run.

| GPU GRES model | `ug` per-user maximum | Initial recommendation |
|---|---:|---|
| `a40` | 2 | One GPU for smoke test / lightweight hand recognition if available |
| `a6000` | 2 | Equivalent first-choice alternative for those tasks |
| `l40` | 2 | Alternative for geometry processing if needed/available |
| `6000ada` | 2 | Alternative; no demonstrated need yet |
| `pro6000` | 2 | Future authorized large-memory inference only if justified |
| `rtx5090` | 1 | Alternative requiring a Blackwell-compatible build |

Use ordinary `ug` QOS for this phase; no need for the killable override.
Actual `sinfo` output, current GPU occupancy, partition details and remaining
billing allowance have not been supplied. No GPU model has been tested.

Current `storagemgr` evidence: SSD allowance 150 GB, assigned 130 GB, including
**`jointposeaction`: 50 GB assigned, 0 B used**. The user created this project.
HDD allowance 300 GB, assigned 200 GB; no dedicated JointPoseAction HDD project.
Other projects are explicitly outside scope: do not inspect, reuse, modify or
clean them. The earlier suggestion to reuse independent project storage is
superseded by this user instruction.

Use only `/projects/jointposeaction`, subject to resolving its SSD tier and
checking permissions. No extra allocation is required to begin. Budget: 15 GB
environment/repo, 5 GB caches/install scratch, 5 GB selected metadata/poses,
10 GB pilot source videos, 5 GB derived outputs, 10 GB free headroom. These are
planning ceilings, not measured dataset sizes. No full archive/video download.

## Next user-executed verification batch (cluster login shell)

```bash
readlink -f /projects/jointposeaction
ls -ld /projects/jointposeaction
test -r /projects/jointposeaction && test -w /projects/jointposeaction && test -x /projects/jointposeaction && printf 'project-access-ok\n'
sinfo -o "%.20N %.6c %.8m %.28G %.10t"
module load Miniforge3/24.11.3-1
conda --version
git --version
git ls-remote https://github.com/shreyaslakhotia/JointPoseAction.git refs/heads/migration/phase1a-readiness
```

These commands allocate nothing and install nothing. Return the output before
running setup jobs. Direct agent noninteractive access remains unavailable;
successful user login does not share that authenticated session with the agent.

## Historical initial connection attempt

## Actual connection evidence

From Windows PowerShell, one noninteractive attempt outside the network sandbox:

```powershell
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15 shreyas010@10.97.216.128 'whoami; hostname; date -u; sinfo -o "%.20N %.6c %.8m %.28G %.10t"; squeue --me; sacctmgr show assoc user=$USER format=Account,QOS%40,DefaultQOS'
```

Exit status: **1**. Relevant SSH output:

```text
EEE/ROSE Cluster 02
Connecting to: login-1

Relevant Guides:
- https://github.com/NTUEEECluster/docs/blob/main/quickstart.md
- https://github.com/NTUEEECluster/docs/blob/main/terms-n-conditions.md
shreyas010@10.97.216.128: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password,keyboard-interactive).
```

The earlier sandbox attempt failed locally with `port 22: Permission denied`;
it was retried outside the sandbox as required. The banner proves reachability,
not authenticated access. `BatchMode=yes` suppresses interactive password prompts;
this failure does not show that the user's password is invalid. Host-key checking
was enabled and no host-key error occurred. No remote discovery command executed.
An unrelated local PowerShell profile execution-policy warning also appeared;
the SSH client nevertheless reached the server. No execution policy was changed.

## Historical ledger before the user's successful login (superseded above)

| Item | Verified state |
|---|---|
| Endpoint | `10.97.216.128:22` reachable |
| Login host | Banner says `login-1`; authenticated `hostname` pending |
| Username | Requested `shreyas010`; authenticated `whoami` pending |
| Partitions, GPU models, node state | Not queried successfully |
| Account, QOS, per-user limits, billing usage | Not queried successfully |
| Current jobs | Not queried successfully |
| Modules, Conda/Python | Not queried successfully |
| Git/GitHub network access from cluster | Not tested |
| Assigned storage, tier, quota, usage, permissions | Not queried successfully |

Historical 100 GB SSD / 200 GB HDD allowances and previously seen GPUs are not
current evidence. No allocation, directory creation, installation, Slurm
submission or dataset transfer occurred.

## Historical first batch — already completed in part

**Windows PowerShell:**

```powershell
ssh -o StrictHostKeyChecking=yes -o ConnectTimeout=15 shreyas010@10.97.216.128
```

Enter authentication only in that terminal. If host-key verification fails,
stop and compare with the current official quick start; do not disable checking.
If a first-time connection needs acceptance, the currently published fingerprint
is `SHA256:AhYlWQBTsN/4GAzYTVTiZzrVhPhcurFMu1sBMglqvdM`.

**Authenticated cluster login shell:** run this small read-only batch:

```bash
whoami
hostname
sinfo -o "%.20N %.6c %.8m %.28G %.10t"
squeue --me
sacctmgr show assoc user="$USER" format=Account,QOS%40,DefaultQOS
module avail
storagemgr
```

In `storagemgr`, inspect allowance, usage and the user's existing allocations.
Do not select create, resize, sharing or deletion actions. Exit using the menu's
displayed exit option; menu keys have not been verified. Return the discovery
output and quota/allocation summary, without credentials. Inspect only assigned
storage. We will explain the menu output before proposing a project location.

The next adaptive read-only batch will inspect the relevant QOS limits, exact
Conda modules, Git/GitHub access and confirmed assigned paths. Avoid listing all
of `/projects`, recursively sizing datasets, or accessing unrelated projects.

## Live policy reviewed before SSH

Reviewed the [documentation repository](https://github.com/NTUEEECluster/docs),
[agent skill](https://github.com/NTUEEECluster/docs/blob/main/skill.md),
[terms](https://github.com/NTUEEECluster/docs/blob/main/terms-n-conditions.md),
[quick start](https://github.com/NTUEEECluster/docs/blob/main/quickstart.md),
[storage](https://github.com/NTUEEECluster/docs/blob/main/storage-guide.md),
[Slurm](https://github.com/NTUEEECluster/docs/blob/main/slurm-guide.md),
[cluster](https://github.com/NTUEEECluster/docs/blob/main/cluster-guide.md) and
[containers](https://github.com/NTUEEECluster/docs/blob/main/container-guide.md).
Live documentation takes precedence over old restore recipes.

Login work is lightweight; installations and processing require compute jobs.
Keep caches/environments on authorized SSD storage, specify one typed GPU model,
and use no prohibited GPU resource flags or scheduling workarounds. New storage,
substantial installs, large downloads and expensive jobs need user approval.

Provisional selection policy, pending actual entitlement: use the least costly
available adequate GPU for pose recognition and small geometry tasks; consider
larger VRAM only for an authorized workload that needs it. The guide lists A40
and A6000 at lower billing weights than L40/6000 Ada or Blackwell. This is not an
account-specific recommendation yet. If Blackwell is selected, the guide requires
a compatible CUDA 12.8+ PyTorch build; exact versions remain undecided.
