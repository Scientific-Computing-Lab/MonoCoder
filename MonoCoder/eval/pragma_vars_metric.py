from metrics import *
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import json


def evaluate_omp(output_file='../monocoder_results.log'):
    '''
    Evaluate inner-clause level classification
    '''
    with open(output_file, 'r') as f:
        labels, preds = [], []
        
        for line in lines:
            sample = json.loads(line.strip())
            labels.append(sample['label'].lower())
            preds.append(sample['pred'].lower())

        # res = {}
        # for label in labels:
        #     d = pragma2dict(label)
        #     if 'reduction' in d:
        #         vs = d['reduction']['vars']
        #         amount = len(vs)
        #         res[amount] = 1 if amount not in res else res[amount]+1

        # print(dict(sorted(res.items())))

        for max_vars in [1,2,3,4,5,100]:
            print(f'# {max_vars}')

            for clause in ['private', 'reduction']:
                preds2, labels2 = [], []

                for label, pred in zip(labels, preds):
                    d = pragma2dict(label)
                    if clause in d:
                        vs = d[clause]['vars']
                        amount = len(vs)
                        
                        if amount <= max_vars:
                            labels2.append(label)
                            preds2.append(pred)

                private_conf = compare_vars(clause, preds2, labels2)
                print(f'{clause} var', private_conf, omp_compute_score(private_conf))

