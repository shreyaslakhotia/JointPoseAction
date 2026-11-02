# Environment setup — prepared, not installed

## Confirmed target and budget

Dedicated project: `jointposeaction`, 50 GB SSD, created by the user. Expected
path `/projects/jointposeaction`; canonical tier/access checks still pending.
Independent projects are outside scope. No HDD allocation is required yet.

Keep the repository at `repo/` and all newly downloaded/processed data in sibling
`data/` directories. No duplicate raw data in the repository. The bootstrap
creates the requested annotations/calibration/pilot/source/processed, environment,
checkpoint, experiment, log and cache directories; it does not allocate storage.

`scripts/cluster/env.sh` checks the dedicated SSD path and configures all six
required cache variables plus Torch/Matplotlib caches. It is sourced by setup and
Slurm scripts, avoiding dependence on interactive `.bashrc` behavior. Scratch is
job-specific; the tiny smoke script does not bulk-delete scratch directories.
No files in other projects or shell startup files are edited.

After verifying storage and cloning the approved migration commit:

```bash
cd /projects/jointposeaction/repo
bash scripts/cluster/create_workspace.sh
```

Prepared source has not been run on the cluster. It fails if the expected SSD
mapping differs; inspect the actual mapping rather than weakening the guard.

## Minimal installation plan — awaiting review and live nodes

Use module `Miniforge3/24.11.3-1` and a prefix environment at
`/projects/jointposeaction/environments/phase1a`, Python 3.12. Install only NumPy,
OpenCV headless, SciPy, pandas, Matplotlib, PyYAML, huggingface_hub, video inspection
utilities and PyTorch. NumPy/SciPy support basic geometry; specialized reconstruction
frameworks are unnecessary. No torchvision/torchaudio unless an actual task needs
them. No model checkpoint downloads.

Package installation must run in a CPU Slurm batch job with caches configured
first; no GPU is needed merely to install wheels. Proposed limit: two CPUs,
6 GiB implicit RAM under the documented CPU policy, 45 minutes. Peak space is
budgeted within 15 GB for environment/repository plus 5 GB scratch/cache; this is
an estimate, not a measured installation. Stop on quota/solver errors and review.

Candidate: PyTorch 2.14.1 CUDA 13.0 wheel, contingent on driver compatibility and
architecture coverage. The official [version-specific instructions](https://pytorch.org/get-started/previous-versions/)
list this build. The general install selector returned an older-looking version
label during this review, so it is not used to infer a version. Cluster guidance
reports driver 595 / CUDA 13.x and requires CUDA 12.8+ for Blackwell; live compute
driver/kernel validation is still required. No local CUDA toolkit install or
compilation is planned; wheels provide runtime dependencies.

Exact scientific package pins must be resolved/reviewed before submitting the
installation job, then captured with `conda list --explicit`, `conda env export`
and `python -m pip freeze` under project logs/specifications. Historical observed
YAMLs are preservation evidence, not this clean environment's specification.
No lockfile, installed version list or successful solver result exists yet.

## Restore and integrity

First verify cluster GitHub access. Clone only this project into the empty
dedicated `repo/`, check out the reviewed migration commit and compare
`git rev-parse HEAD`. Do not overwrite an existing checkout or bypass GitHub
authentication. If HTTPS cloning fails, transfer a source-only Git bundle through
the local machine; no dataset relay through Git. That fallback is not yet needed.

No upstream clone is needed for the GPU smoke test. AssemblyHands metadata/toolkit
restoration will use its pinned revision when needed. SAM 2 and SAM 3D remain
deferred. Run the preservation checker in a CPU batch job after cloning. Log YAML
parsing separately and retain exact source commit and job exit status.

Current result: no cluster clone, environment installation, prefix activation,
GPU execution, package lock or cache-path execution has been verified.
