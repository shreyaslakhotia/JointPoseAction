#!/usr/bin/env bash
# One login command; all substantive work runs under Slurm.
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
"$JPA_PYTHON" -m phase1c.pipeline submit "$@"
