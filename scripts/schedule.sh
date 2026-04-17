#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=aether
#SBATCH --mem=100G
#SBATCH --time=100
#SBATCH --output=logs/out_%j.out
#SBATCH --error=logs/err_%j.err

# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# Variables
# shellcheck disable=SC1091
source .env

#Environment
# shellcheck disable=SC1091
source .venv/bin/activate

# Runs
#srun python src/train.py experiment=alignment trainer=$TRAINER_PROFILE logger=$LOGGER
srun python -u src/train.py experiment=alignment_v1

# example runs with overwritten configs
#srun python src/train.py experiment=alignment trainer=ddp_sim trainer.max_epochs=10 data.pin_memory=false
