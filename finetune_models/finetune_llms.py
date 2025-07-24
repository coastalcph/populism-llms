import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from peft import LoraConfig, get_peft_model, TaskType
from data import DATA_DIR
from data.helpers import augment_dataset
from audit_llms.helpers import SYSTEM_PROMPT, INSTRUCTION, QUERY, RESPONSE_START
from datasets import load_dataset
import logging, datasets
import sys
from trl import DataCollatorForCompletionOnlyLM


logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return f"trainable params: {trainable_params} || all params: {all_param} || " \
           f"trainable%: {100 * trainable_params / all_param}"


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='Qwen/Qwen3-4B', help='Model name in HF Hub')
    parser.add_argument('--dataset_name', default='coastalcph/populism-trump-2016', help='Dataset name in HF Hub')
    parser.add_argument('--upsample_ratio', default=5, type=int, help="The upsampling ratio for the train dataset.")
    parser.add_argument('--epochs', default=3, type=int, help='Number of epochs to train')
    parser.add_argument('--per_device_train_batch_size', default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, default=False, help='Whether to use debug mode')
    param_config = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # param_config.debug = True
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if param_config.debug:
        print('Debugging mode activated')
        tokenizer_name = param_config.model_name
        param_config.model_name = 'gpt2'
        param_config.quant = False
        param_config.max_length = 8
    else:
        tokenizer_name = param_config.model_name

    # Report configuration parameters
    print('Configuration parameters:')
    for arg in vars(param_config):
        print(f'{arg}: {getattr(param_config, arg)}')

    # Compute free memory for each GPU
    if torch.cuda.is_available():
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
        max_memory = f"{free_in_GB - 2}GB"
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
    else:
        max_memory = None

    # Load tokenizer and model
    model = AutoModelForCausalLM.from_pretrained(
        param_config.model_name,
        max_memory=max_memory,
        device_map='auto' if torch.cuda.is_available() else 'cpu',
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ) if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare LORA model
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    # Cast the output to float32
    model.lm_head = CastOutputToFloat(model.lm_head)

    # Set the LORA config
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules if not param_config.debug else None,
    )

    # Init PEFT model
    model = get_peft_model(model, config)

    # Report the number of trainable parameters
    print(print_trainable_parameters(model))

    # Load the dataset
    LABELS = {0: '(a) No populism',
              1: '(b) Anti-elitism',
              2: '(c) People-centrism',
              3: '(d) Both people-centrism and anti-elitism'}

    dataset = load_dataset(param_config.dataset_name)['train']

    if param_config.upsample_ratio:
        dataset = augment_dataset(dataset, upsample_ratio=param_config.upsample_ratio)

    def map_to_instructions(example):
        if len(example['sentence'].split(' ')) > 48:
            example['sentence'] = ' '.join(example['sentence'].split(' ')[:48])
        annotation_request = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": INSTRUCTION +
                                                      QUERY.format(example['sentence'])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

        # add response
        annotation_request += RESPONSE_START
        annotation_request += LABELS[example['pop_code']][1:]
        example['text'] = annotation_request

        return example

    # Turn text into instruction
    dataset = dataset.map(map_to_instructions, load_from_cache_file=False)
    print('Demonstrating the first 10 samples:')
    for i in range(10):
        print(dataset[i]['text'])
        print('-' * 100)

    # Tokenize the dataset
    dataset = dataset.shuffle(seed=param_config.seed)
    dataset = dataset.map(lambda samples: tokenizer(samples["text"], padding="max_length",
                                                    truncation=True, max_length=300), batched=True,
                          load_from_cache_file=False)

    dataset = dataset.remove_columns(['sentence', 'pop_code'])

    if 'meta-llama' in param_config.model_name:
        response_template = '<|start_header_id|>assistant<|end_header_id|>'
    elif 'Qwen' in param_config.model_name:
        response_template = '</think>'

    # Prepare the dataset for training
    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=param_config.per_device_train_batch_size,
            gradient_accumulation_steps=param_config.gradient_accumulation_steps,
            per_device_eval_batch_size=param_config.per_device_train_batch_size,
            num_train_epochs=param_config.epochs,
            optim="paged_adamw_32bit",
            warmup_ratio=0.1,
            weight_decay=0.001,
            max_grad_norm=1.0,
            learning_rate=param_config.lr,
            lr_scheduler_type="cosine",
            fp16=True if torch.cuda.is_available() else False,
            logging_strategy="steps",
            log_level="info",
            logging_first_step=True,
            save_total_limit=5,
            logging_steps=100,
            save_strategy="epoch",
            output_dir=os.path.join(DATA_DIR, 'finetuned_models', f'{param_config.model_name.split("/")[1]}-up-{param_config.upsample_ratio}'),
            seed=param_config.seed,
        ),
        data_collator=DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()
