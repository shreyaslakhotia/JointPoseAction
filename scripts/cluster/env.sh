#!/usr/bin/env bash
# Source from both login setup and Slurm jobs. Do not edit ~/.bashrc automatically.
jpa_configure_paths() {
    local resolved
    resolved=$(readlink -f /projects/jointposeaction) || return 1
    if [[ "$resolved" != /projects/_ssd/jointposeaction || ! -d "$resolved" || ! -w "$resolved" ]]; then
        printf 'Expected writable dedicated SSD /projects/_ssd/jointposeaction; got %s\n' "$resolved" >&2
        return 1
    fi
    export JPA_ROOT=/projects/jointposeaction
    export JPA_ENV="$JPA_ROOT/environments/phase1a"
    export TMPDIR="$JPA_ROOT/cache/tmp/${SLURM_JOB_ID:-login}"
    export PIP_CACHE_DIR="$JPA_ROOT/cache/pip"
    export CONDA_PKGS_DIRS="$JPA_ROOT/cache/conda"
    export HF_HOME="$JPA_ROOT/cache/huggingface"
    export TRITON_CACHE_DIR="$JPA_ROOT/cache/triton"
    export XDG_CACHE_HOME="$JPA_ROOT/cache/xdg"
    export TORCH_HOME="$JPA_ROOT/cache/torch"
    export MPLCONFIGDIR="$JPA_ROOT/cache/matplotlib"
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$HF_HOME" \
        "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"
}
jpa_configure_paths
