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
| Submission / job ID | Not submitted / none |
| Environment | Not installed |
| GPU / CUDA operation | Not tested |
| Output/error logs | None |
| Exit status | No execution |

Local syntax checks do not validate Bash/module behavior, node drivers, kernel
architecture support or Slurm acceptance. All remain pending actual execution.
