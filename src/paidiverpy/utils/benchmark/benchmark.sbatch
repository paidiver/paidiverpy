#!/usr/bin/env bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --chdir=__WORKDIR__
#SBATCH --export=ALL
#SBATCH --partition=__QUEUE__
__ACCOUNT_DIRECTIVE__
#SBATCH --time=__WALLTIME__
#SBATCH --output=benchmark-paidiverpy-%j.out
#SBATCH --error=benchmark-paidiverpy-%j.err
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEMORY__GB

set -euo pipefail

__ACTIVATION__

exec paidiverpy -bt __BENCHMARK_PARAMS__ -c __CONFIG_FILE__