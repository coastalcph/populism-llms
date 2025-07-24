#!/bin/bash
#SBATCH --job-name=finetune-llms-qwen
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=2:00:00

# Model Parameters
MODEL_PATH='Qwen/Qwen3-14B'

export PYTHONPATH=.

python ./finetune_llms/finetune_llms.py \
  --model_name ${MODEL_PATH} \
  --text_unit sentences \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --upsample_ratio 5