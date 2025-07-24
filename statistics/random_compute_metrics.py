import json
import os.path
import argparse
import random
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
    parser.add_argument('--filename', default='gemini_few-shot-128-gemini-thinking=0.csv', help='Model name in HF Hub')
    parser.add_argument('--text_unit', default='sentences', type=str, help='Text unit of input')
    parser.add_argument('--dataset_name', default='trump', type=str, help='Dataset name')
    config = parser.parse_args()

    y_true = []
    y_pred = []

    dataset = load_dataset(config.dataset_name, text_unit=config.text_unit)
    print()
    if config.dataset_name == 'trump':
        dataset = split_dataset(dataset)['validation']

    for sample in dataset:
        pred = random.choices([0, 1, 2, 3], weights=[0.92, 0.04, 0.02, 0.02], k=1)[0]
        if pred == 3:
            pred = [1, 2]
        elif pred in [0, 1, 2]:
            pred = [pred]
        label = int(sample['label'])
        if label == 3:
            label = [1, 2]
        elif label in [0, 1, 2]:
            label = [label]
        y_pred.append(multi_labels_to_ids(pred))
        y_true.append(multi_labels_to_ids(label))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Compute metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    f1_class = f1_score(y_true, y_pred, average=None)

    print(f'MODEL: {config.filename}, #0: {f1_class[0]:.3f} #1: {f1_class[1]:.3f} #2 {f1_class[2]:.3f} Macro: {macro_f1:.3f}')


if __name__ == '__main__':
    main()