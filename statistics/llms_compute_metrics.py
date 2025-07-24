import json
import os.path
import argparse
import re
from data import DATA_DIR
from sklearn.metrics import classification_report

LETTER_LABELS = {'a': [1, 0, 0], 'b': [0, 1, 0], 'c': [0, 0, 1], 'd': [0, 1, 1]}
NUMBER_LABELS = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1], 3: [0, 1, 1]}
FULL_LABELS = {'no populism': [1, 0, 0], 'elitism': [0, 1, 0], 'people': [0, 0, 1], 'both': [0, 1, 1]}


def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--filename', default='Qwen3-14B_trump_k_0_seed_42_options_1_responses.jsonl', help='Model name in HF Hub')
    config = parser.parse_args()

    y_true = []
    y_pred = []

    for k in [0]:
        with open(os.path.join(DATA_DIR, 'model_responses', config.filename.format(k)), 'r') as f:
            for line in f:
                data = json.loads(line)
                response = data['model_response']
                if ' (a)' in response:
                    json_dict = {'label': 'a'}
                elif ' (b)' in response:
                    json_dict = {'label': 'b'}
                elif ' (c)' in response:
                    json_dict = {'label': 'c'}
                elif ' (d)' in response:
                    json_dict = {'label': 'd'}

                if json_dict['label'] in LETTER_LABELS:
                    y_pred.append(LETTER_LABELS[json_dict['label']])
                else:
                    for option in FULL_LABELS:
                        if re.search(option, json_dict['label'], re.IGNORECASE):
                            y_pred.append(FULL_LABELS[option])
                            break
                y_true.append(NUMBER_LABELS[data['label']])

        print(classification_report(y_true, y_pred, target_names=['No', 'AE', 'PC'], digits=3))



if __name__ == '__main__':
    main()