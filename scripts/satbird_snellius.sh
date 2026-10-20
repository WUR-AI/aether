#!/bin/bash
##SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_a100
##SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=aether
#SBATCH --time=6:00:00
#SBATCH --output=/gpfs/home2/gtijunaityte/aether/logs/%j_out.out
##SBATCH --error=/gpfs/home2/gtijunaityte/aether/logs/%j_err.err
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=scratch-node

echo "[runner.sh] Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "[runner.sh] Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

# Define the cleanup/sync-back function
cleanup_and_sync() {
    if [ "$FLAG" = true ]; then
        echo "========= TRAP TRIGGERED: Syncing results back to home directory ========="
        rsync -avP --ignore-existing ./logs/ /gpfs/home2/gtijunaityte/aether/logs/
        rsync -avP --ignore-existing ./data/checkpoints/ /gpfs/home2/gtijunaityte/aether/data/checkpoints/

        # rm -r logs/*
        # rm -r src/*
    fi
}
trap cleanup_and_sync EXIT INT TERM

source .env
FLAG=false
if [ "$PROJECT_ROOT" != "." ]; then
    FLAG=true
    echo "Ensuring project root directory exists: $PROJECT_ROOT"

    # update project files
    EXCLUDE_DATA=("tessera" "tessera_v1.0" "s2" "aef")
    RSYNC_EXCLUDES_DATA=()
    for item in "${EXCLUDE_DATA[@]}"; do
        RSYNC_EXCLUDES_DATA+=(--exclude="eo/$item")
    done

    echo "Updating project files, excluding specific eo/ subfolders:"
    echo "${RSYNC_EXCLUDES_DATA[@]}"
    rsync -avP \
        "${RSYNC_EXCLUDES_DATA[@]}" \
        --exclude '.venv' \
        --exclude 'docs' \
        --exclude 'data/checkpoints' \
        --exclude 'uv.lock' \
        --exclude 'logs' \
        --exclude 'data/cache/' \
        --exclude '.github' \
        --exclude 'notebooks' \
        ./ "$PROJECT_ROOT/"

    # sync data
    EXCLUDE_SUBFOLDERS=("${EXCLUDE_DATA[@]}")
    RSYNC_EXCLUDES=()
    for item in "${EXCLUDE_SUBFOLDERS[@]}"; do
        RSYNC_EXCLUDES+=(--exclude="eo/$item")
    done
    echo "Syncing data directory and ignoring:"
    echo "${RSYNC_EXCLUDES[@]}"
    rsync -avP --ignore-existing  "${RSYNC_EXCLUDES[@]}" "data/satbird-USA-summer/" "$PROJECT_ROOT/data/satbird-USA-summer/"
    rsync -avP --ignore-existing data/cache/ "$PROJECT_ROOT/data/cache/"

    # Move to the project root for execution
    cd "$PROJECT_ROOT"
    echo "Changed working directory to: $(pwd)"
fi

# Create venv on node-local $TMPDIR; keep code + data on scratch (PROJECT_ROOT).
ml 2025
export UV_PROJECT_ENVIRONMENT="$TMPDIR/.venv"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"

uv sync --directory "$PROJECT_ROOT"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
uv pip install -e "$PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Runs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Starting model training..."


# AVR AEF
#srun python -u src/train.py experiment=s2bms_prediction/avr_aef_128_256_deeper_mlp
srun python -u src/train.py experiment=s2bms_prediction/avr_aef_128_256_lin
srun python -u src/train.py experiment=s2bms_prediction/avr_aef_128_256_mlp

# GEOCLIP
srun python -u src/train.py experiment=s2bms_prediction/geoclip_deeper_mlp
srun python -u src/train.py experiment=s2bms_prediction/geoclip_lin
srun python -u src/train.py experiment=s2bms_prediction/geoclip_mlp

# SATCLIP
srun python -u src/train.py experiment=s2bms_prediction/satclip_deeper_mlp
srun python -u src/train.py experiment=s2bms_prediction/satclip_lin
srun python -u src/train.py experiment=s2bms_prediction/satclip_mlp

# BASELINES
#srun python -u src/train.py experiment=s2bms_prediction/baselines/tabular_linear
#srun python -u src/train.py experiment=s2bms_prediction/baselines/tabular_mlp
