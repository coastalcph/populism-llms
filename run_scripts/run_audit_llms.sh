#!/bin/bash
#SBATCH --job-name=audit-llms-qwen
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=0:20:00

# Model Parameters
MODEL_PATH='Qwen/Qwen3-14B'

export PYTHONPATH=.

python ./audit_llms/audit_llms.py \
  --model_name ${MODEL_PATH} \
  --max_length 2 \
  --k_shot 0
