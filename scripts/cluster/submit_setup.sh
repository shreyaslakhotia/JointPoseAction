#!/usr/bin/env bash
# User-run login-node submission only. No installation or GPU work here.
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
bash scripts/cluster/create_workspace.sh
source scripts/cluster/env.sh
if [[ -e "$JPA_ENV" ]]; then
    printf 'Environment exists; inspect installation state before submitting again.\n' >&2
    exit 1
fi
mkdir -p "$JPA_ROOT/experiments/setup"
# Atomic guard prevents accidental double submission. Failed/completed setup
# must be reviewed before clearing this marker; never auto-resubmit.
marker="$JPA_ROOT/experiments/setup/submitted"
mkdir "$marker" || { printf 'Setup already submitted; inspect recorded jobs.\n' >&2; exit 1; }
install_reply=$(sbatch --parsable scripts/cluster/install_environment.sbatch)
install_id=${install_reply%%;*}
[[ "$install_id" =~ ^[0-9]+$ ]] || { printf 'Unexpected sbatch response\n' >&2; exit 1; }
printf '%s\n' "$install_id" > "$marker/install-job-id.txt"
printf 'Installation job: %s\n' "$install_id"
smoke_reply=$(sbatch --parsable --gres=gpu:l40:1 \
    --dependency="afterok:$install_id" --kill-on-invalid-dep=yes scripts/cluster/gpu_smoke.sbatch)
smoke_id=${smoke_reply%%;*}
[[ "$smoke_id" =~ ^[0-9]+$ ]] || { printf 'Unexpected sbatch response\n' >&2; exit 1; }
printf '%s\n' "$smoke_id" > "$marker/smoke-job-id.txt"
printf 'Dependent L40 smoke job: %s\n' "$smoke_id"
printf 'Logs: %s/logs/jpa-install-%s.{out,err} and jpa-smoke-%s.{out,err}\n' "$JPA_ROOT" "$install_id" "$smoke_id"
