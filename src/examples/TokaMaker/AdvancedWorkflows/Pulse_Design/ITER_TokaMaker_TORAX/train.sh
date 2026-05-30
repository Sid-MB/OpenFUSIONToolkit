#!/bin/bash

export WANDB_API_KEY=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
# Load micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate iql

# Run training
python IQL.py