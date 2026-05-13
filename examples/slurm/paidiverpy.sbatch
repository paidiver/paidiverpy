#!/usr/bin/env bash
#SBATCH --job-name=paidiverpy-driver
#SBATCH --chdir=/home/users/tobfer/paidiver/paidiverpy
#SBATCH --export=ALL
#SBATCH --partition=standard
#SBATCH --account=paidiver
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --qos=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB

set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate PaidiverpyNew

exec paidiverpy -c "/home/users/tobfer/paidiver/paidiverpy/examples/config_files/config_benthic_client_slurm.yml"
