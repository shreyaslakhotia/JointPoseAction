#!/usr/bin/env bash
# Source after env.sh. Never trust an inherited Conda activation/PATH in Slurm.
: "${JPA_ENV:?Source scripts/cluster/env.sh first}"
export JPA_PYTHON="$JPA_ENV/bin/python"
if [[ ! -x "$JPA_PYTHON" ]]; then
    printf 'Project interpreter is missing or not executable: %s\n' "$JPA_PYTHON" >&2
    return 1
fi
unset PYTHONHOME PYTHONPATH
export PATH="$JPA_ENV/bin:$PATH"
"$JPA_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import sys

expected = Path(os.environ["JPA_ENV"]).resolve()
if Path(sys.prefix).resolve() != expected:
    raise RuntimeError(f"Wrong Python prefix: {sys.prefix}; expected {expected}")
import numpy
print(json.dumps({"project_python": sys.executable, "prefix": sys.prefix,
                  "numpy": numpy.__version__}), flush=True)
PY
