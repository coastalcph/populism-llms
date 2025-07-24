#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import numpy as np
from sklearn.metrics import f1_score
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback
)
from transformers import set_seed as tf_set_seed
from datasets import load_dataset
from data.helpers import augment_dataset
from accelerate.utils import set_seed as acc_set_seed
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed=13):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    acc_set_seed(seed)
    tf_set_seed(seed)
    torch.backends.cudnn.deterministic = True


LABELS = {0: '(a) No populism',
          1: '(c) Anti-elitism populism',
          2: 'c) People-centrism populism'}

SIMPLIFIED_LABELS = {0: '(a) No populism',
                     1: '(b) Populism'
                     }

label_list = ['(a) No populism',
              '(b) Anti-elitism populism',
              '(c) People-centrism populism']

simplified_label_list = ['(a) No populism', '(b) Populism']

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_seq_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    text_unit: str = field(
        default="sentences",
        metadata={
            "help": "The text unit to use for training."
        }
    )
    dataset_name: str = field(
        default="coastalcph/populism-trump-2016",
        metadata={"help": "TDataset name in HF Hub"}
    )
    split_train_dataset: bool = field(
        default=True, metadata={"help": "Whether to split dataset for hyper-parameter tuning."}
    )
    shuffle_train_dataset: bool = field(
        default=True, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    augment_train_dataset: bool = field(
        default=True, metadata={"help": "Whether to augment the train dataset or not."}
    )
    upsample_ratio: int = field(
        default=5, metadata={"help": "The upsampling ratio for the train dataset."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Init Seed
    seed_everything(data_args.custom_seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    raw_datasets = load_dataset(data_args.dataset_name)
    if data_args.split_train_dataset:
        raw_datasets_train = raw_datasets['train'].train_test_split(test_size=0.2)
        raw_datasets['train'] = raw_datasets_train['train']
        raw_datasets['validation'] = raw_datasets_train['validation']
        raw_datasets['test'] = raw_datasets_train['test']

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2 if data_args.binary_task else 3,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
    )
    config.problem_type = "multi_label_classification"

    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    for param in model.parameters():
        param.data = param.data.contiguous()

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    else:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[int]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label] = 1.0
        return ids

    def preprocess_function(examples):
        # Tokenize the texts
        if data_args.text_unit == "sentences":
            result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        elif data_args.text_unit == "windows":
            result = {'input_ids': [], 'attention_mask': []}
            for example_pars in examples['sentence']:
                ex_result = tokenizer(example_pars, padding=False, max_length=63, truncation=True, add_special_tokens=False)
                input_ids = [tokenizer.cls_token_id]
                input_ids += ex_result['input_ids'][4] + [tokenizer.pad_token_id] * (64 - len(ex_result['input_ids'][4])) + [tokenizer.sep_token_id]
                attention_mask = [1]
                attention_mask += [1] * len(ex_result['input_ids'][4])
                attention_mask += [0] * (64 - len(ex_result['input_ids'][4])) + [1]
                for idx, sent_input_ids in enumerate(ex_result['input_ids'][:3] + ex_result['input_ids'][4:]):
                        input_ids += sent_input_ids
                        attention_mask += [1] * len(sent_input_ids)
                input_ids += [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (511 - len(input_ids))
                attention_mask += [1] + [0] * (511 - len(attention_mask))
                ex_result['input_ids'] = input_ids
                ex_result['attention_mask'] = attention_mask
                result['input_ids'].append(ex_result['input_ids'])
                result['attention_mask'].append(ex_result['attention_mask'])

            # Fix labels
            result["label"] = [[l] if l != 3 else [1, 2] for l in examples["pop_code"]]
            result["label"] = [multi_labels_to_ids(l) for l in result["label"]]

        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        for split in raw_datasets:
            if split == 'train':
                if data_args.augment_train_dataset:
                    raw_datasets[split] = augment_dataset(raw_datasets[split], data_args.upsample_ratio)
            raw_datasets[split] = raw_datasets[split].map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=42)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            eval_dataset = eval_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding

        # F1 scores is commonly used in multi-label classification
        micro_f1 = f1_score(y_pred=preds, y_true=p.label_ids, average="micro")
        macro_f1 = f1_score(y_pred=preds, y_true=p.label_ids, average="macro")

        return {'micro_f1': micro_f1, 'macro_f1': macro_f1}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=None)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        predictions = trainer.predict(eval_dataset, metric_key_prefix="eval").predictions
        predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])

        output_predict_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Eval results *****")
                if data_args.binary_task:
                    writer.write("index\tpredictions\tlabel\n")
                else:
                    writer.write("index\tpredictions\tlabels\n")
                for index, item in enumerate(predictions):
                    if data_args.binary_task:
                        item = item
                        if data_args.text_unit == 'paragraphs':
                            gold = [1 if l != 0 else 0 for l in eval_dataset['label'][index]]
                        else:
                            gold = 1 if eval_dataset['label'][index] != 0 else 0
                    else:
                        if data_args.text_unit == 'paragraphs':
                            # recover from multi-hot encoding
                            item = [[i for i in range(len(par_item)) if par_item[i] == 1] for par_item in item]
                            gold = [[i for i in range(len(par_labels)) if par_labels[i] == 1] for par_labels in eval_dataset['label'][index]]
                        else:
                            # recover from multi-hot encoding
                            item = [i for i in range(len(item)) if item[i] == 1]
                            gold = [i for i in range(len(eval_dataset['label'][index])) if eval_dataset['label'][index][i] == 1]
                    writer.write(f"{index}\t{item}\t{gold}\n")
        logger.info("Predict results saved at {}".format(output_predict_file))

    # Test
    if training_args.do_predict:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["predict_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        predictions = trainer.predict(eval_dataset, metric_key_prefix="predict").predictions
        predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])

        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Test results *****")
                writer.write("index\tpredictions\tlabels\n")
                for index, item in enumerate(predictions):
                    # recover from multi-hot encoding
                    item = [i for i in range(len(item)) if item[i] == 1]
                    gold = [i for i in range(len(eval_dataset['label'][index])) if eval_dataset['label'][index][i] == 1]
                    writer.write(f"{index}\t{item}\t{gold}\n")
        logger.info("Predict results saved at {}".format(output_predict_file))


if __name__ == "__main__":
    main()
