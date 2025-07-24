import re
from transformers import set_seed as tf_set_seed
from transformers import AutoTokenizer, AutoProcessor
from sentence_transformers import SentenceTransformer
import transformers
import torch
import numpy as np
import tqdm
import os
import json
import argparse
import sys
import random
import copy
from peft import PeftModel
from helpers import (SYSTEM_PROMPT, RESPONSE_START, QUERY, DEMONSTRATION_EXAMPLES, INSTRUCTION)
from data import DATA_DIR
from datasets import load_dataset
from accelerate.utils import set_seed as acc_set_seed
from transformers.utils import logging

logging.set_verbosity_error()


def build_rag(corpus_dataset):
    """Initialize the retriever with embeddings"""
    print("Initializing retriever...")
    # Use a lightweight sentence transformer model
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Generate embeddings for the corpus
    print("Generating embeddings...")
    corpus_embeddings = embedding_model.encode(
        corpus_dataset['text'],
        show_progress_bar=True,
        convert_to_tensor=True
    ).cpu().numpy()

    return embedding_model, corpus_embeddings


def seed_everything(seed=13):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    acc_set_seed(seed)
    tf_set_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--model_name', default='Qwen/Qwen3-32B', help='Model name in HF Hub')
    parser.add_argument('--dataset_name', default='coastalcph/populism-trump-2016', type=str, help='Dataset name')
    parser.add_argument('--use_prior_context',  default=False, type=bool, help='Whether to use prior context')
    parser.add_argument('--lora_adapter', default=None, help='LoRa adapter')
    parser.add_argument('--max_length', default=2, type=int, help='Maximum length of the generated text')
    parser.add_argument('--seed', default=42, type=int, help='Seed for reproducibility')
    parser.add_argument('--k_shot', default=0, type=int, help='Number of K shots given as context to the model')
    parser.add_argument('--pipeline', action=argparse.BooleanOptionalAction, help='Whether to use hf pipeline or model.generate')
    parser.add_argument('--distribution_awareness', action=argparse.BooleanOptionalAction, help='Whether to share the distribution')
    parser.add_argument('--retrieval_augmentation', default=False, type=bool, help='Whether to share the distribution')
    parser.add_argument('--k_retrieved_docs', default=8, type=int, help='Whether to share the distribution')
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction, help='Whether to use debug mode')
    config = parser.parse_args()
    seed_everything(seed=config.seed)

    IDS_TO_LABELS = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    IDS_TO_FULL_LABELS = {0: '(a) No populism', 1: '(b) Anti-elitism', 2: '(c) People-centrism', 3: '(d) Both people-centrism and anti-elitism'}

    # Load dataset
    dataset = load_dataset(config.dataset_name)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    # If few-shot prompting, collect samples per category (class)
    if config.k_shot > 0:
        no_pop_subset = train_dataset.filter(lambda el: el['pop_code'] == 0 and 8 < len(el['sentence'].split(' ')) < 32)
        no_pop_subset.shuffle(seed=config.seed)
        no_pop_samples = no_pop_subset.select(range(int(config.k_shot/4)))
        pop_samples = []
        for label in range(1, 4):
            pop_subset = train_dataset.filter(lambda el: el['pop_code'] == label and 8 < len(el['sentence'].split(' ')) < 32)
            pop_subset.shuffle(seed=config.seed)
            pop_samples.extend([sample for sample in pop_subset.select(range(int(config.k_shot / 4)))])
        k_shot_samples = pop_samples + [sample for sample in no_pop_samples]
        random.shuffle(k_shot_samples)
    else:
        config.k_shot_mode = 0

    # Gather prior context (5 sentences)  per examined sentence
    if config.use_prior_context:
        before = []
        context_data = []
        last_speech_id = ''
        for idx, sample in enumerate(eval_dataset):
            if sample['speech_id'] != last_speech_id:
                before = []
            if len(before) == 6:
                before.pop(0)
            context_data.append(copy.deepcopy(before))
            before.append(sample['sentence'])
            last_speech_id = sample['speech_id']

    # Build RAG and extract relevant documents from training set
    if config.retrieval_augmentation:
        embedding_model, corpus_embeddings = build_rag(train_dataset)
        retrieved_documents = []
        print("Generating embeddings...")
        query_embeddings = embedding_model.encode(
            dataset['sentence'],
            show_progress_bar=True,
            convert_to_tensor=True
        ).cpu().numpy()
        for idx, example in tqdm.tqdm(enumerate(dataset)):
            # Compute cosine similarity between query and corpus
            similarities = embedding_model.similarity(query_embeddings[idx], corpus_embeddings).cpu().numpy()
            indices = np.argsort(similarities[0])[::-1][:config.k_retrieved_docs]

            # Return the top K documents
            retrieved_documents.append([f"{idx + 1}. \"{train_dataset['sentence'][doc_idx]}\" - "
                                        f"Category ({['a', 'b', 'c', 'd'][int(train_dataset['pop_code'][doc_idx])]})"
                                        for idx, doc_idx in enumerate(indices)])
    if config.debug:
        print('Debugging mode activated')
        tokenizer_name = config.model_name
        config.model_name = 'gpt2'
        config.max_length = 8
    else:
        tokenizer_name = config.model_name

    # Load tokenizer, if Gemma 3 AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if 'google/gemma-3-27b-it' in config.model_name:
        processor = AutoProcessor.from_pretrained(config.model_name)
    else:
        processor = False

    # Compute free memory for each GPU
    if torch.cuda.is_available():
        free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
        max_memory = f"{free_in_GB - 2}GB"
        n_gpus = torch.cuda.device_count()
        max_memory = {i: max_memory for i in range(n_gpus)}
    else:
        max_memory = None

    # Load model
    print('Loading + Quantizing model...')
    output_name = config.model_name.split('/')[-1]
    model_config = transformers.AutoConfig.from_pretrained(
        config.model_name,
        token=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            use_flash_attention=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        ) if config.debug is False else None,
        device_map='auto',
        token=True,
        max_memory=max_memory if not config.debug else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    # Load LoRA, if enabled
    if config.lora_adapter:
        output_name += '_lora'
        model = PeftModel.from_pretrained(model, config.lora_adapter,
                                          device_map="auto",
                                          max_memory=max_memory if not config.debug else None)
        print('Loaded LoRA adapter...')

    # SetUp HuggingFace pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        processor=processor
    )

    # Output filename expansion
    if config.distribution_awareness is True:
        output_name += '_dist'

    if config.use_prior_context is True:
        output_name += '_use_prior'

    if config.retrieval_augmentation:
        output_name += '_rag'

    # Create output file
    f_out = open(os.path.join(DATA_DIR, "model_responses", f"{output_name}_k_{config.k_shot}_seed_{config.seed}_options_responses.jsonl"), "w")

    # Iterate over the examples in the evaluation dataset and save the responses
    examples = []
    for idx, example in tqdm.tqdm(enumerate(dataset)):
        compute_xai = True if config.xai else False
        if 'gemma' in tokenizer_name:
            if config.k_shot == 0 and config.k_shot_mode == 0:
                if config.text_unit == 'sentences':
                    annotation_request = tokenizer.apply_chat_template(
                        conversation=[{"role": "system", "content": [{"type": "text", "text":SYSTEM_PROMPT}]},
                                      {"role": "user",
                                       "content": [
                                           {"type": "text", "text": INSTRUCTION + '\n\n' + QUERY.format(example['sentence'])}
                                       ]
                                       }
                                      ],
                        tokenize=False, add_generation_prompt=True)
        else:
            if config.retrieval_augmentation:
                CONTEXT = '\n'.join(retrieved_documents[idx])
                CONTEXT = f'Here are the most similar {config.k_retrieved_docs} sentences from the training set, accompanied by their label:\n\n' + CONTEXT
                CONTEXT = CONTEXT + '\n\nWhen classifying a sentence, focus primarily on the content of that specific sentence.\n\n'
                annotation_request = tokenizer.apply_chat_template(
                    conversation=[{"role": "system", "content": SYSTEM_PROMPT if config.system_prompt else ''},
                                  {"role": "user",
                                   "content": INSTRUCTION + '\n\n' + CONTEXT + '\n\n' + QUERY.format(example['sentence'])}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            elif config.use_prior_context:
                CONTEXT = '\n'.join([f'- {sentence}' for idx, sentence in enumerate(context_data[idx])])
                if len(CONTEXT) == 0:
                    CONTEXT = 'There are no preceding sentences. The following sentence is the very first sentence of this speech.\n\n'
                else:
                    CONTEXT = 'Here are the preceding sentences for context:\n\n' + CONTEXT
                CONTEXT = CONTEXT + ('\n\nWhen classifying a sentence, focus primarily on the content of that specific sentence. '
                           'Use the context of preceding sentences only to resolve coreferences (e.g., identifying who "they" or "you" refer to) '
                           'or to disambiguate when the sentence is ambiguous on its own.\n\n')
                annotation_request = tokenizer.apply_chat_template(
                    conversation=[{"role": "system", "content": SYSTEM_PROMPT if config.system_prompt else ''},
                                  {"role": "user", "content": INSTRUCTION + '\n\n' + CONTEXT + '\n\n' + QUERY.format(example['sentence'])}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            elif config.k_shot == 0 and config.k_shot_mode == 0:
                annotation_request = tokenizer.apply_chat_template(
                    conversation=[{"role": "system", "content": SYSTEM_PROMPT if config.system_prompt else ''},
                                  {"role": "user", "content": INSTRUCTION + '\n\n' + QUERY.format(example['sentence'])}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                if config.k_shot_mode == 0:
                    conversation = [{"role": "system", "content": SYSTEM_PROMPT if config.system_prompt else ''}]
                    conversation.append({"role": "user", "content": INSTRUCTION + '\n\n' + QUERY.format(k_shot_samples[0]['sentence'])})
                    conversation.append({"role": "assistant", "content": RESPONSE_START + IDS_TO_LABELS[k_shot_samples[0]['pop_code']] + '"}\n```'})
                    for sample in k_shot_samples[1:]:
                        conversation.append({"role": "user", "content": QUERY.format(sample['sentence'])})
                        conversation.append({"role": "assistant", "content": RESPONSE_START + IDS_TO_LABELS[sample['pop_code']] + '"}\n```'})
                    conversation.append({"role": "user", "content": QUERY.format(example['sentence'])})
                    annotation_request = tokenizer.apply_chat_template(conversation=conversation, tokenize=False,
                                                                       add_generation_prompt=True,  enable_thinking=False)
                else:
                    conversation = [{"role": "system", "content": SYSTEM_PROMPT if config.system_prompt else ''}]
                    k_shot_dict = {}
                    for sample in k_shot_samples:
                        if sample['label'] not in k_shot_dict:
                            k_shot_dict[sample['pop_code']] = [sample['sentence']]
                        else:
                            k_shot_dict[sample['pop_code']].append(sample['sentence'])

                    k_examples = ''
                    for label in range(4):
                        k_examples += DEMONSTRATION_EXAMPLES.format(IDS_TO_FULL_LABELS[label])
                        k_examples += '\n'.join([f'- {example}' for example in k_shot_dict[label]]) + '\n\n'
                    conversation.append({"role": "user", "content": INSTRUCTION + '\n\n' + k_examples +
                                                                    QUERY.format(example['sentence'])})
                    annotation_request = tokenizer.apply_chat_template(conversation=conversation, tokenize=False,
                                                                       add_generation_prompt=True,  enable_thinking=False)

        annotation_request = re.sub('\n{3,}', '\n\n', annotation_request)

        if config.distribution_awareness:
            distribution_text = 'The label distribution is (a) No populism (92%), (b) Anti-elitism (4%), (c) People-centrism (2%), (d) Both people-centrism and anti-elitism (2%).'
            annotation_request = annotation_request.replace('Which is the most', distribution_text + '\n\n' + 'Which is the most')

        # Print out the instruction
        print('-' * 150)
        print('REQUEST:', annotation_request)
        print('-' * 150)

        # Get the response from the chatbot
        responses = pipeline(
            annotation_request,
            do_sample=False,
            num_return_sequences=1,
            return_full_text=False,
            use_cache=False,
            max_new_tokens=config.max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )

        # Print the response
        print("-" * 50)
        print(f'RESPONSE:\n{RESPONSE_START}{responses[0]["generated_text"]}')
        print("-" * 50)

        # Save the response
        try:
            example["model_response"] = RESPONSE_START + responses[0]['generated_text']
        except:
            print('RESPONSE: None\n')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(f'Error: {exc_type} at line {exc_tb.tb_lineno}')
            # Save the response
            example["response"] = None

        # Save the response
        f_out.write(json.dumps(example) + "\n")

    # Close output file
    f_out.close()


if __name__ == '__main__':
    main()
