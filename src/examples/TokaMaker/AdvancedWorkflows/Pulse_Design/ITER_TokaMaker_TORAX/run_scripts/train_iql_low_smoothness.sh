#!/usr/bin/env bash
# train_iql_low_smoothness.sh — Train an IQL actor with reduced smoothing bias.
#
# This is an explicit ablation wrapper for checking whether the default
# residual-action + previous-action setup is over-regularizing the policy.
# It keeps the training pipeline unchanged, but turns off the action-rate
# penalty and uses a less history-coupled observation mode by default.
#
# Use this when:
#   - you want to test whether the current policy is too conservative
#   - you want a fast comparative run against the default training setup
#   - you are willing to trade some smoothness for more expressive control
#
# Default: OBSERVATION_MODE=plasma_only (drops action history from the observation).
# IQL auto-selects ACTION_MODE=absolute for plasma_only, and the action-rate penalty
# only applies in residual mode, so this path is intentionally less smooth.
#
# Example:
#   DATASET_DIR=./run_prev_action_smoke_20260530_222521 \
#     NUM_STEPS=2000 CHECKPOINT_INTERVAL=1000 EVAL_INTERVAL=200 \
#     sbatch run_scripts/train_iql_low_smoothness.sh

set -euo pipefail

export OBSERVATION_MODE="${OBSERVATION_MODE:-plasma_only}"

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/IQL.py" ]; then
  PROJECT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
fi

exec bash "${PROJECT_DIR}/run_scripts/train_iql.sh" "$@"
