#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=jag-standard
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Optional setup
# sh ./setup-env.sh

N_WORKERS="${N_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

uv run --extra cuda13 python collect_trajectories_delta.py \
  --n_trajectories 1000 \
  --start_idx 600 \
  --n_workers "${N_WORKERS}" \
  --output_dir ./rl_dataset_delta_sampling_maxloop=2_grid_51_run1 \
  "$@"
