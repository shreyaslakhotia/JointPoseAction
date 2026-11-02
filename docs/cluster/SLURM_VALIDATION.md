# Slurm validation — prepared, not submitted

Phase 1B: live L40 availability/model and canonical SSD path were supplied.
Use `bash scripts/cluster/submit_setup.sh` for CPU installation plus a dependent
one-L40 smoke test. See `PHASE1B_EXECUTION.md`. No fresh hardware discovery is
required. Actual execution ledger below is unchanged until jobs finish.

The user association is `ug-proj`, default QOS `ug`. One GPU is sufficient.
Live node/GRES output is still missing, so no model is silently selected.

Prepared files: `scripts/cluster/gpu_smoke.sbatch`, `gpu_validate.py`, `env.sh`.
The GPU script deliberately has no CPU/memory flags. The submission must name
exactly one verified GPU model. It has a five-minute time limit, ordinary `ug`
QOS and separate absolute output/error paths. Create `logs/` before submission.

After the environment installation succeeds and a model is verified, submit
once from `/projects/jointposeaction/repo`, replacing the model token:

```bash
sbatch --parsable --gres=gpu:VERIFIED_MODEL:1 scripts/cluster/gpu_smoke.sbatch
```

This is a template, not a command to run with the literal token. Prefer `a40` or
`a6000` if live availability permits. No killable override is needed. If Slurm
rejects the submission, retain the message and stop; do not vary flags to bypass
the rejection. Scheduler permission is not established by QOS table entries alone.

The test prints Python, NumPy, OpenCV and PyTorch versions, build CUDA version,
compiled architectures, CUDA availability, job ID, GPU identity/capability and
numerical result. It reads the project upstream manifest and checks a small PNG
read/write roundtrip in project storage. It imports no model code and trains
nothing. A failed assertion raises an exception and produces a nonzero exit.

Collect the returned job ID, both logs and accounting after completion:

```bash
sacct -X -j JOB_ID --format=JobID,State,ExitCode,Elapsed,NodeList -P
```

Record actual log filenames, `COMPLETED` state and `0:0` exit status along with
the successful JSON result; an empty error file alone is insufficient evidence.

## Actual execution ledger

| Field | Result |
|---|---|
| Installation job | 241924; user accounting `COMPLETED`, `0:0` |
| Environment | Install-script checks passed by exit status; exact exports/logs awaiting review |
| GPU smoke job | 241925, `COMPLETED`, 25 seconds |
| GPU / CUDA operation | User log: NVIDIA L40S, capability 8.9, driver CUDA API 13000; tensor and project I/O passed |
| Output/error logs | User supplied `jpa-smoke-241925.out`; status passed, `.err` empty |
| GPU exit status | `0:0` |

Runtime log versions: Python 3.12.16, NumPy 2.2.6, OpenCV 4.12.0,
PyTorch 2.14.1+cu130, CUDA runtime 13.0. Compiled architecture list includes
sm_75/80/86/90/100/120; actual L40S kernel operation succeeded. CLI repair
242207 completed `0:0` in six seconds; private HF login then succeeded.
CPU inspection 242209 failed before downloads with a base-Python NumPy import
error. E0 scripts now select the absolute project interpreter and verify its
prefix rather than relying on inherited activation. No new GPU job was submitted.

Local syntax checks do not validate Bash/module behavior, node drivers, kernel
architecture support or Slurm acceptance. All remain pending actual execution.
