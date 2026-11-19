#!/bin/bash
# =============================================================================
# Narval SLURM job: fine-tune MuscleMimic (mm-motion-100) on the boxing clips.
# 1 x A100, 10 hours. Submit from a LOGIN node, from your WORK folder:
#     cd /lustre06/project/6086787/deborah && sbatch narval_boxing/jobs/train_boxing_a100.sh
# Teammates: change WORK below to your own folder (see the setup guide).
# =============================================================================
#SBATCH --job-name=boxing_finetune
#SBATCH --account=def-durandau
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=deborah.sinishaw@mail.mcgill.ca     # change it to whoever is running the job

set -euo pipefail

# --- Your folders (one per teammate) -------------------------------------------
WORK=/lustre06/project/6086787/deborah         # <-- teammates: your own folder
REPO=$WORK/musclemimic_mcgill_dev
BUNDLE=$WORK/narval_boxing
CKPT=$WORK/pretrained/mm-motion-100

# --- Software ------------------------------------------------------------------
module load StdEnv/2023 python/3.11
# Do NOT `module load cuda`: jax[cuda12] brings its own CUDA libraries.

# --- Compute nodes are offline: everything was downloaded on the login node ----
export CONVERTED_AMASS_PATH=$BUNDLE/clips      # -> clips/MyoFullBody/gmr_v6/skills/*.npz
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$WORK/wandb
export MUJOCO_GL=egl
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# JAX (network) and MuJoCo Warp (physics) share the GPU. JAX reserves this fraction up front;
# 0.90 left Warp too little and crashed with 'Warp CUDA error 2: out of memory'.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
export PYTHONUNBUFFERED=1                     # write print() output to the log immediately

mkdir -p "$WORK/logs" "$WANDB_DIR"
cd "$REPO"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
git log -1 --format='repo commit: %h %s'

# .venv/bin/python, not `uv run`: uv run would try to re-sync over the network.
.venv/bin/python -m fullbody.experiment \
    --config-dir="$BUNDLE/configs" \
    --config-name=conf_boxing_finetune \
    experiment.resume_from="$CKPT"

echo "Finished $(date)"
