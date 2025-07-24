from os.path import join, isdir
from tqdm import tqdm
from os import makedirs
from datasets import load_dataset
import pandas as pd
import torch
from torch import nn, Tensor
import numpy as np
from xai_utils.tokenization_utils import merge_roberta_tokens
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from xai.xai_utils.xai_roberta import RobertaForSequenceClassificationXAI
from xai.xai_utils.xai_utils import plot_conservation, compute_lrp_explanation
from typing import Any
import argparse


class Zero(nn.Identity):
    """A layer that just returns Zero-Embeddings"""

    def __init__(self, dim=768, *args: Any, **kwargs: Any) -> None:
        self.dim = dim
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.zeros((input.shape[0], input.shape[1], self.dim)).to(input.device)

def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--modelname', default='../roberta-large', help='Model name in HF Hub')
    parser.add_argument('--dataset_name', default='trump', help='Dataset name')
    parser.add_argument('--dataset_split', default='test', help='Dataset name')
    parser.add_argument('--text_unit', default='sentences', help='Dataset mode')
    parser.add_argument('--res_folder', default='./results')
    param_config = parser.parse_args()

    modelname = param_config.modelname
    dataset_name = param_config.dataset_name
    dataset_split = param_config.dataset_split

    DEBUG=False

    if 'roberta-small' in modelname:
        model_case = 'roberta-small'
        model_class = RobertaForSequenceClassificationXAI
    elif 'roberta-base' in modelname:
        model_case = 'roberta-base'
        model_class = RobertaForSequenceClassificationXAI
    elif 'roberta-large' in modelname:
        model_case = 'roberta-large'
        model_class = RobertaForSequenceClassificationXAI
    else:
        raise NotImplementedError

    filename_out = f"relevance_{dataset_split}.pkl"
    filename_out = filename_out

    res_folder = join(param_config.res_folder, modelname)

    if not isdir(res_folder):
        makedirs(res_folder)

    print(join(res_folder, filename_out))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if DEBUG:
        config = AutoConfig.from_pretrained(
            "FacebookAI/roberta-base",
            num_labels=3,
            finetuning_task="text-classification",
        )
        config.problem_type = "multi_label_classification"
        tokenizer = AutoTokenizer.from_pretrained(
            "FacebookAI/roberta-base"
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-base",
            config=config,
        )
        for param in model.parameters():
            param.data = param.data.contiguous()

    else:
        if param_config.machine == 'hendrix':
            config = AutoConfig.from_pretrained(join("/projects/nlp/data/data/populism", modelname))
            model = AutoModelForSequenceClassification.from_pretrained(
                join("/projects/nlp/data/data/populism", modelname), config=config)
        elif param_config.machine == 'karolina':
            config = AutoConfig.from_pretrained(join("/scratch/project/eu-25-45/populism/models", modelname))
            model = AutoModelForSequenceClassification.from_pretrained(
                join("/scratch/project/eu-25-45/populism/models", modelname), config=config)
        else:
            config = AutoConfig.from_pretrained(modelname)
            model = AutoModelForSequenceClassification.from_pretrained(modelname, config=config)
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    # Load explainable model

    model_xai = model_class(model.config, lrp=True)
    return_gradient_norm = False
    state_dict_ = model.state_dict()

    _ = model_xai.load_state_dict(state_dict_)
    _ = model.load_state_dict(state_dict_)

    model_xai.eval()
    model_xai.to(device)

    model.eval()
    model.to(device)

    if dataset_name == 'trump':
        labels_dict = {0: '(a) No populism',
                       1: '(b) Anti-elitism',
                       2: '(c) People-centrism',
                       3: '(d) Both people-centrism and anti-elitism'}
        dataset = load_dataset(dataset_name)[dataset_split]
        labels = labels_dict.keys()
        key = 'text'
        prediction_dict = {(1, 0, 0): 0,
                           (0, 1, 0): 1,
                           (0, 0, 1): 2,
                           (0, 1, 1): 3,
                           (1, 1, 0): 4,
                           (1, 0, 1): 4,
                           (1, 1, 1): 4,
                           (0, 0, 0): 5,}
    else:
        raise NotImplementedError

    generation_ids_dict = None

    def preprocess_function(examples, padding, examples_key):
        # Tokenize the texts
        # Padding and truncation switched off
        batch = tokenizer(
            examples[examples_key],
            padding=padding,
            max_length=128,
            truncation=True,
        )
        return batch

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=None,
        load_from_cache_file=False,
        fn_kwargs={"padding": False,
                   "examples_key": key
                   },
    )

    df = pd.DataFrame(columns=['id', 'tokens', 'attention', 'y_pred', 'y_true', 'logits', 'data'])

    Rs = []
    Ls = []

    # Bypass embeddings
    if 'roberta' in model_case:
        token_embeddings = model_xai.roberta.embeddings
        model_xai.roberta.embeddings = nn.Identity()

        model_components = {'embeddings': token_embeddings,
                            'encoder': model_xai.roberta.encoder,
                            'classifier': model_xai.classifier}

    df_ii = 0

    for ii in tqdm(range(len(tokenized_dataset))):

        data = tokenized_dataset[ii]
        inputs_ = data['input_ids']
        tokens = tokenizer.convert_ids_to_tokens(inputs_)
        inputs_ = torch.tensor(inputs_).unsqueeze(0)


        example_id = ii
        y_true = tokenized_dataset[ii]['label']

        try:
            output = model(input_ids=inputs_.to(device), output_attentions=False)
        except:
            print(f'skip datapoint {ii}: {tokens}')
            continue
        logits = output['logits'].squeeze().detach().cpu()
        predictions = np.array([np.where(p > 0, 1, 0) for p in logits])
        y_pred = prediction_dict[tuple(predictions)]

        if y_pred == 3:
            xai_loop = [1, 2]
        elif y_pred == 5:
            xai_loop = []
        elif y_pred == 4:
            xai_loop = [x for x in range(len(predictions)) if predictions[x]==1]
        else:
            xai_loop = [y_pred]

        for y_pred in xai_loop:
            model_inputs = {'input_ids': inputs_.to(device)}
            logit_func = lambda x: x[:, y_pred]

            try:
                relevance, selected_logit, logits = compute_lrp_explanation(model_components,
                                                                            model_inputs,
                                                                            logit_function=logit_func,
                                                                            model_case=model_case,
                                                                            return_gradient_norm=return_gradient_norm,
                                                                            tokenizer=tokenizer,
                                                                            generation_ids_dict=generation_ids_dict)
            except AssertionError:
                print(f'skip datapoint {ii}: {tokens}')
                continue

            Ls.append(selected_logit)
            Rs.append(relevance.sum())

            df.loc[df_ii] = [example_id, tokens, relevance, y_pred, y_true, logits, data]
            df_ii += 1

            tokens_merged, relevance_merged = merge_roberta_tokens(tokens, relevance)

            r_normalization = np.max(np.abs(relevance_merged))
            relevance_normalized = np.array(relevance_merged) / r_normalization


    df.to_pickle(join(res_folder, filename_out))
    filename_conservation = "conservation.png"
    plot_conservation(Ls, Rs, join(res_folder, filename_conservation))


if __name__ == '__main__':
    main()
