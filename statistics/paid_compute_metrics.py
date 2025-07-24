import json
import os.path
import argparse
import re

import tqdm

from data import DATA_DIR
from sklearn.metrics import f1_score, precision_score, recall_score
from data.dataset import load_dataset, split_dataset
from typing import List
import numpy as np

def multi_labels_to_ids(labels: List[int]) -> List[int]:
    ids = [0, 0, 0]
    for label in labels:
        ids[label] = 1
    return ids

def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--filename', default='openai_zero-shot-gpt-4.1-thinking=0.csv', help='Model name in HF Hub')
    parser.add_argument('--text_unit', default='sentences', type=str, help='Text unit of input')
    parser.add_argument('--dataset_name', default='trump', type=str, help='Dataset name')
    config = parser.parse_args()

    y_true = []
    y_pred = []
    y_true_god = []
    sentences = []

    with open('/Users/rwg642/Desktop/TEST-god-mode.tsv', 'r') as file:
        for idx, line in enumerate(file):
            if idx == 0:
                continue
            god_label = eval(line.strip().split('\t')[4])
            if god_label == 3:
                god_label = [1, 2]
            else:
                god_label = [god_label]
            y_true_god.append(multi_labels_to_ids(god_label))
            sentences.append(line.strip().split('\t')[3])

    dataset = load_dataset(config.dataset_name, text_unit=config.text_unit)
    print()
    if config.dataset_name == 'trump':
        dataset = split_dataset(dataset)['validation']

    with open(os.path.join(DATA_DIR, 'model_responses', config.filename), 'r') as f:
        for idx, line in tqdm.tqdm(enumerate(f)):
            if idx == 0:
                continue
            sample = line.strip().split('\t')
            text = sample[3]
            pred = int(sample[4])
            if pred == 3:
                pred = [1,2]
            elif pred in [0,1,2]:
                pred = [pred]
            elif pred == 4:
                pred = [1,2]

            if text == dataset['text'][idx-1]:
                label = int(dataset['label'][idx-1])
                if label == 3:
                    label = [1, 2]
                elif label in [0, 1, 2]:
                    label = [label]
                elif label == 4:
                    label = [1, 2]
                y_pred.append(multi_labels_to_ids(pred))
                y_true.append(multi_labels_to_ids(label))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true_god = np.array(y_true_god)

    # Compute metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    f1_class = f1_score(y_true, y_pred, average=None)

    print(f'MODEL: {config.filename}, #0: {f1_class[0]:.3f} #1: {f1_class[1]:.3f} #2 {f1_class[2]:.3f} Macro: {macro_f1:.3f}')


if __name__ == '__main__':
    main()