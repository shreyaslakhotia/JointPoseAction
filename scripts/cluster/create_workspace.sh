#!/usr/bin/env bash
set -euo pipefail
umask 077
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$JPA_ROOT"/{data/{source,annotations,calibration,pilot,processed},environments,checkpoints,experiments,logs}
printf 'Workspace: %s\nEnvironment target: %s\nScratch: %s\n' "$JPA_ROOT" "$JPA_ENV" "$TMPDIR"
