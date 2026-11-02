# Resource discovery — 2 November 2026

Status: cluster network reachable; authentication and resource discovery pending.

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

## Verification ledger

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

## Next batch — user terminal

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
