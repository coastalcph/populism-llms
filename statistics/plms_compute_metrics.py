import os.path
import argparse

from data import DATA_DIR
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
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
    parser.add_argument('--model_name', default='roberta-large', help='Model name')
    config = parser.parse_args()

    y_true = []
    y_pred = []

    with open(os.path.join(DATA_DIR, 'finetuned_models', config.model_name, 'predict_results.txt'), 'r') as f:
        for index, line in enumerate(f):
            if index == 0:
                continue
            values = line.split('\t')
            y_true.append(multi_labels_to_ids(eval(values[2])))
            y_pred.append(multi_labels_to_ids(eval(values[1])))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    report = classification_report(y_true, y_pred, labels=range(3), target_names=['None', 'Elite', 'People'], digits=3)
    print(report)


if __name__ == '__main__':
    main()
