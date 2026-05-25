#!/usr/bin/env bash

#SBATCH --account=nlp
#SBATCH --gres=gpu:1
#SBATCH --constraint=48G
#SBATCH --mem=128G
#SBATCH --partition=jag-standard
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Optional setup
# sh ./setup-env.sh

uv run python collect_trajectories_delta.py --n_trajectories 1000 --start_idx 600 --output_dir ./rl_dataset_delta_sampling_maxloop=2_grid_51_run1 "$@"
