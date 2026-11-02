#!/usr/bin/env bash
# Login submission only: attach preparation, then queue one GPU job and CPU evaluation.
set -euo pipefail
umask 077
cd /projects/jointposeaction/repo
source scripts/cluster/env.sh
source scripts/cluster/use_environment.sh
"$JPA_PYTHON" -m e0.pipeline submit "$@"
