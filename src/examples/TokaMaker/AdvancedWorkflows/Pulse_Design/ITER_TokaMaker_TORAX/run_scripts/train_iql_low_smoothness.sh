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
# Defaults chosen here are intentionally less smooth than the main training path:
#   ACTION_RATE_PENALTY=0
#   OBSERVATION_MODE=plasma_only
#   ACTION_MODE=absolute
#
# Example:
#   DATASET_DIR=./run_prev_action_smoke_20260530_222521 \
#     NUM_STEPS=2000 CHECKPOINT_INTERVAL=1000 EVAL_INTERVAL=200 \
#     sbatch run_scripts/train_iql_low_smoothness.sh

set -euo pipefail

export ACTION_RATE_PENALTY="${ACTION_RATE_PENALTY:-0}"
export OBSERVATION_MODE="${OBSERVATION_MODE:-plasma_only}"
export ACTION_MODE="${ACTION_MODE:-absolute}"

exec "$(dirname "$0")/train_iql.sh" "$@"
