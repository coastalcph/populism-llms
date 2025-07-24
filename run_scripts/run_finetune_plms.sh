#!/bin/bash
#SBATCH --job-name=finetune-plms-roberta
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --time=2:00:00


export HUGGINGFACE_HUB_CACHE=/scratch/project/eu-25-45/it4i-ihalk/huggingface

# Model Parameters
MODEL_PATH='FacebookAI/roberta-large'
export TRANSFORMERS_VERBOSITY=error
export PYTHONPATH=.

TRANSFORMERS_VERBOSITY=error python ./finetune_llms/finetune_plms.py \
  --model_name ${MODEL_PATH} \
  --text_unit sentences \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 10 \
  --metric_for_best_model macro_f1 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --upsample_ratio 5 \
  --output_dir ../finetuned_models/roberta-large \
  --custom_seed 42 \
  --do_train \
  --do_predict \
  --evaluation_strategy epoch \
  --save_strategy epoch