from metrics import *
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import json


def evaluate_omp(output_file='../compcoder_results.jsonl'):
    '''
    Evaluate clause level classification
    '''
    with open(output_file, 'r') as f:
        labels, preds = [], []
        
        for line in lines:
            sample = json.loads(line.strip())
            labels.append(sample['label'].lower())
            preds.append(sample['prediction'].lower())

        print('private', compare_directive('private', preds, labels))
        print('reduction', compare_directive('reduction', preds, labels))

        # print('private var', compare_vars('private', preds, labels))
        # print('reduction var', compare_vars('reduction', preds, labels))
        print('reduction operator', compare_vars('reduction', preds, labels, operator=True))


def plot_bar(result: dict, metric='precision', output_file=None):
    labels, values = [], []

    for k, v in result.items():
        labels.append(k)
        values.append(omp_compute_score(v, metric=metric))
    
    plt.figure(figsize=(12,8))
    plt.bar(labels, values)
    plt.ylabel(metric)
    plt.title('OMP Pragma Generation Eval')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if not output_file:
        plt.show()
    else:
        plt.savefig(output_file)


def plot_confusion_matrix(ax, cm, title):
    sns.heatmap([[cm['TP'], cm['FP']], [cm['FN'], cm['TN']]], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)


def plot_conf_marices(result: dict):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    for (k, cm), ax in zip(result.items(), axs.flatten()):
        plot_confusion_matrix(ax, cm, title=k)

    fig.delaxes(axs[1, 2]) 
    plt.tight_layout()    

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

