#!/usr/bin/env bash
#SBATCH --job-name=paidiverpy-driver
#SBATCH --output=paidiverpy-driver-%j.out
#SBATCH --error=paidiverpy-driver-%j.err

set -euo pipefail

# Submit this script with site-specific Slurm flags, for example:
# sbatch --partition=standard --account=paidiver --time=12:00:00 \
#   --export=ALL,PAYDIVERPY_ENV=PaidiverpyNew \
#   deployment/slurm/paidiverpy.sbatch examples/config_files/config_benthic_client_slurm.yml

CONFIG_FILE="${1:-examples/config_files/config_benthic_client_slurm.yml}"
REPO_ROOT="${PAYDIVERPY_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PAYDIVERPY_ENV="${PAYDIVERPY_ENV:-}"
PAYDIVERPY_EXECUTABLE="${PAYDIVERPY_EXECUTABLE:-}"

cd "$REPO_ROOT"

if command -v micromamba >/dev/null 2>&1 && [[ -n "$PAYDIVERPY_ENV" ]]; then
    eval "$(micromamba shell hook -s bash)"
    micromamba activate "$PAYDIVERPY_ENV"
elif command -v conda >/dev/null 2>&1 && [[ -n "$PAYDIVERPY_ENV" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$PAYDIVERPY_ENV"
fi

if [[ -n "$PAYDIVERPY_EXECUTABLE" ]]; then
    exec "$PAYDIVERPY_EXECUTABLE" -c "$CONFIG_FILE"
fi

if ! command -v paidiverpy >/dev/null 2>&1; then
    echo "paidiverpy is not available in the current Slurm job environment." >&2
    echo "Set PAYDIVERPY_ENV or activate the environment before submitting, or export PAYDIVERPY_EXECUTABLE." >&2
    exit 1
fi

exec paidiverpy -c "$CONFIG_FILE"
