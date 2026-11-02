# Environment setup — CPU/GPU checks passed; inspection interpreter fix prepared

## Explicit interpreter selection after inspection failure

GPU test 241925 and CLI repair 242207 both completed `0:0`; user logs confirm
L40S execution, Python 3.12.16, Torch 2.14.1+cu130, CUDA driver API 13000,
NumPy 2.2.6 and OpenCV 4.12.0. Authentication succeeded on project storage.
Inspection job 242209 failed importing NumPy from the module/base Python before
downloads. Inherited activation followed by module loading is a plausible PATH
cause; the exact shell transition has not been observed. Do not reinstall NumPy
into the base environment or rebuild the validated project environment.

All E0 job entry points now source `use_environment.sh` and invoke the absolute
`$JPA_ENV/bin/python`. The helper removes inherited PYTHONHOME/PYTHONPATH, checks
the actual Python prefix and imports/logs NumPy before running tests or downloads.
It does not activate Conda, install packages, change Slurm flags or submit GPU jobs.
The failed job did not reach the metadata downloader. Resubmission uses the
same CPU resource request and approved data scope after updating the source.

## Targeted HF CLI repair

The user reported `hf auth login` failed importing `click` after updating to
`ab393c1`. Login did not occur and `set -e` prevented submission of the inspection
job. Initial installation did not test CLI startup; its successful import/ABI
checks therefore did not establish that the `hf` command worked. The direct
requirements now pin `click==8.1.8`, and fresh installs check `hf auth login --help`.

For the existing environment, `repair_hf_cli.sbatch` installs only Click (98,188
wheel bytes), with no dependency changes, in a one-CPU, five-minute job. It runs
`pip check`, tests the real authentication help command without authenticating,
and preserves before/after package lists, install report and updated export.
No Torch/CUDA package, GPU allocation or dataset download is included. This
repair has been syntax-checked locally but needs actual cluster execution.

## Confirmed target and budget

Dedicated project: `jointposeaction`, 50 GB SSD, created by the user. User terminal
output confirms `/projects/jointposeaction` resolves to
`/projects/_ssd/jointposeaction`; bootstrap will check directory writability.
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

## Phase 1B installation procedure — executable, not yet run

The user supplied live nodes: `l40`, `a40`, `a6000`, `6000ada`, `pro6000` and
`rtx5090` match permitted QOS models. L40 had a mixed node in that snapshot;
availability may change. Queue one L40 smoke test normally; no hardware benchmark
or wait for an idle preferred model. The idle GH200 is not established as a
permitted GPU by the user's QOS evidence and is not selected.

`install_environment.sbatch` installs on a two-CPU compute allocation (45-minute
limit), then `submit_setup.sh` submits a dependent five-minute, one-L40 test.
The GPU is allocated only after CPU setup succeeds. Invalid dependencies are
cancelled. Submission errors stop the helper; inspect any installation job
already submitted rather than trying alternate scheduler flags. A persistent
submission marker prevents accidental duplicate jobs.

The clean prefix remains `environments/phase1a` to reuse the existing smoke-test
contract; its historical name does not imply a second environment is required.
Existing/partial prefixes are never overwritten automatically.

Direct scientific/data pins are in
`reproducibility/environments/e0-requirements.txt`. Python is constrained to 3.12
and PyTorch to 2.14.1 from the official `cu130` index. Scientific requirements use
PyPI and wheel-only installation. NumPy 2.2.6 is deliberately below 2.3, satisfying
OpenCV 4.12's NumPy constraint for Python 3.12. NumPy, SciPy, pandas, OpenCV and
PyTorch CPU bridge imports are checked together, followed by `pip check` and the
separate GPU kernel check. No torchvision is needed. No compiler or system CUDA
installation is performed. Versioned package metadata was inspected on PyPI.

Compatibility basis: the current cluster guide reports driver 595 supporting
CUDA 13.x, and the official PyTorch version page lists 2.14.1/cu130. The selected
L40 is supported by this CUDA generation. The actual node's CUDA driver API
version and GPU capability are logged during the smoke test, and a real kernel
operation must succeed. Documentation is compatibility evidence, not a substitute
for that execution result.

Installation records under `logs/environment-JOB_ID/` include source commit,
requirements, pip installation reports, source-integrity output, Conda explicit
specification/environment export and pip freeze. Those exports pin the resolved
Python patch and transitive dependencies; direct pins alone are not a full lock.
Nothing has been installed on the cluster by preparing these files.

From a clean cluster clone of the reviewed E0 branch:

```bash
bash scripts/cluster/submit_setup.sh
```

This installs the minimal environment and submits the dependent GPU check.
Expected peak space remains approximately 15 GB environment/repository plus
5 GB installation scratch, not measured yet. Pip wheel caching is disabled to
avoid retaining duplicate large CUDA downloads; all other caches remain on SSD.
Retain failures and partial state for diagnosis. No dataset/model downloads run
as part of setup.

## Historical planning notes (superseded by the executable procedure above)

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
