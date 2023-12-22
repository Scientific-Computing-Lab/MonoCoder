
from collections import Counter
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from bleu.google_bleu import avg_bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge


def ir_metrics(references, candidates):
    """
    An ir metrics for binary classification.

    Args:
        references (list[int]): A list of references, each reference should be 0 or 1
        candidates (list[int]): A list of candidates, each candidate should be 0 or 1

    Returns:
        dict[str, int]: Mapping metric names to scores

    """
    p = precision_score(y_true=references, y_pred=candidates)
    r = recall_score(y_true=references, y_pred=candidates)
    f1 = f1_score(y_true=references, y_pred=candidates)
    return {'precision': p, 'recall': r, 'f1': f1}


def __ir_metrics(references, candidates):
    """
    An ir metrics for a list of references and candidates, each of both is a single token so that need exact match.

    Args:
        references (list[str]): A list of references, each reference should be one token
        candidates (list[str]): A list of candidates, each candidate should be one token

    Returns:
        (float, float, float):
            - precision
            - recall
            - f1

    """
    p, r, f1 = 0, 0, 0
    if len(references) == 0:
        if len(candidates) == 0:
            p, r, f1 = 1, 1, 1
    else:
        common = Counter(candidates) & Counter(references)
        num_same = sum(common.values())
        if num_same != 0:
            p = 1.0 * num_same / len(candidates)
            r = 1.0 * num_same / len(references)
            f1 = (2 * p * r) / (p + r)
    return p, r, f1


def avg_ir_metrics(references, candidates):
    """
    Calculate precision, recall and f1 score,
    this version of ir metrics calculate the avg scores of each candidate in candidates.

    Args:
        references (list[list[str]]): A list of references, each reference should be tokenized into a list of tokens
        candidates (list[list[str]]): A list of candidates, each candidate should be tokenized into a list of tokens

    Returns:
        dict: Dict of mapping ir metric names to scores

    """
    total_p, total_r, total_f1 = 0, 0, 0
    for reference, candidate in zip(references, candidates):
        p, r, f1 = __ir_metrics(references=reference, candidates=candidate)

        total_p += p
        total_r += r
        total_f1 += f1

    size = len(references)
    return {'avg_precision': total_p / size, 'avg_recall': total_r / size, 'avg_f1': total_f1 / size}


def remove_white_characters(s):
    """
    Join the list of tokens and remove all white characters.

    Args:
        s (Union(list[str], str)): List of tokens

    Returns:
        str: String after removing white characters

    """
    if isinstance(s, list):
        s = ''.join(s)
    s = s.lower()
    s = re.sub(r'\s', '', s)
    return s


def accuracy_for_sequence(references, candidates):
    """
    Calculate accuracy,
        this version of ir metrics calculate scores of each candidate in candidates
        which match the corresponding reference exactly (except white characters).

    Args:
        references (list[list[str]]): A list of references, each reference should be a list of tokens
        candidates (list[list[str]]): A list of candidates, each candidate should be a list of tokens

    Returns:
        dict[str, float]: Dict of mapping ir metric names to scores

    """
    references = [remove_white_characters(reference) for reference in references]
    candidates = [remove_white_characters(candidate) for candidate in candidates]
    return accuracy(references=references, candidates=candidates)


def accuracy(references, candidates):
    """
    Calculate accuracy.

    Args:
        references (list): A list of references, each reference should be an object
        candidates (list): A list of candidates, each candidate should be an object

    Returns:
        dict[str, float]: Dict of mapping ir metric names to scores

    """
    acc = accuracy_score(y_true=references, y_pred=candidates)
    return {'accuracy': acc}


def accuracy_top_k_for_sequence(references, candidates):
    """
    Calculate accuracy,
        this version of ir metrics calculate scores of each candidate in candidates
        which match the corresponding reference exactly (except white characters).

    Args:
        references (list[str]): A list of references, each reference should be a string
        candidates (list[list[str]]): A list of candidates, each candidate should be a list of string

    Returns:
        dict[str, float]: Dict of mapping ir metric names to scores

    """
    references = [remove_white_characters(reference) for reference in references]
    candidates = [[remove_white_characters(can) for can in candidate] for candidate in candidates]
    return accuracy_top_k(references=references, candidates=candidates)


def accuracy_top_k(references, candidates):
    """
    Calculate accuracy for top k.

    Args:
        references (list): A list of references, each reference should be an object
        candidates (list[list]): A list of candidates, each candidate should be a list of k candidates

    Returns:
        dict[str, float]: Dict of mapping ir metric names to scores

    """
    k = len(candidates[0])
    acc_top_k = {}
    for i in range(1, k + 1):
        acc_top_k[f'accuracy_top_{i}'] = 0
    for reference, candidate_list in zip(references, candidates):
        for i in range(1, k + 1):
            candidate_set = set(candidate_list[:i])
            if reference in candidate_set:
                acc_top_k[f'accuracy_top_{i}'] += 1
    for key, value in acc_top_k.items():
        acc_top_k[key] = value / len(references)
    return acc_top_k


def bleu(references, candidates):
    """
    Calculate avg BLEU.

    Args:
        references (list[list[str]]): A list of references, each reference should be tokenized into a list of tokens
        candidates (list[list[str]]): A list of candidates, each candidate should be tokenized into a list of tokens

    Returns:
        dict[str, float]: Dict of mapping metric name and avg score

    """
    return {'bleu': avg_bleu(references=references, candidates=candidates)}


def meteor(references, candidates):
    """
    Calculate avg METEOR.

    Args:
        references (list[list[str]]): A list of references, each reference should be tokenized into a list of tokens
        candidates (list[list[str]]): A list of candidates, each candidate should be tokenized into a list of tokens

    Returns:
        dict[str, float]: Dict of mapping metric name and avg score

    """
    meteor_calculator = Meteor()
    return {'meteor': meteor_calculator.compute_score(references=references, candidates=candidates)[0]}


def rouge_l(references, candidates):
    """
    Calculate avg ROUGE-L.

    Args:
        references (list[list[str]]): A list of references, each reference should be tokenized into a list of tokens
        candidates (list[list[str]]): A list of candidates, each candidate should be tokenized into a list of tokens

    Returns:
        dict[str, float]: Dict of mapping metric name and avg score

    """
    rouge_calculator = Rouge()
    return {'rouge-l': rouge_calculator.compute_score(references=references, candidates=candidates)[0]}


def omp_valid_paren(pragma):
    '''
    Balanced non-canonical parentheses
    '''
    count = 0

    for ch in pragma:
        if count < 0 or count > 1:
            return False

        if ch == '(':
            count += 1
        elif ch == ')':
            count -= 1

    return count == 0


def omp_valid_pragma(pragma):
    clause_vars = ['private', 'reduction', 'collapse', 'schedule', 'shared', 'default', 'aligned', 'num_threads']

    if not omp_valid_paren(pragma):
        return False

    # parentheses after clause
    paren_indexes = [idx for idx, ch in enumerate(pragma) if ch =='(']
    for idx in paren_indexes:
        if all([not pragma[:idx].rstrip().endswith(clause) for clause in clause_vars]):
            return False

    # clause before parentheses
    for clause in clause_vars[:2]:
        idx = pragma.find(clause)

        if idx == -1: continue

        if not pragma[idx+len(clause):].lstrip().startswith('('):
            return False

        if clause == 'reduction':
            reduction_part = pragma[idx+len(clause):]
            reduction_part = reduction_part[reduction_part.find('('):reduction_part.find(')')]

            if reduction_part.count(':') == 0:
                return False

    return True


def pragma2dict(pragma):
    """
    Convert an openmp pragma into dictionary.
    do private ( var_501 )  ->   {private.vars: [var_501]}

    Assumes legal omp pragmas
    """
    result = {}
    pattern = r' (private)\s*\((.+?)\)| (reduction)\s*\((.+?):(.+?)\)| (simd)'
    pragma = pragma.replace(',', ' ')

    matches = re.findall(pattern, pragma)
    for match in matches:
        private, private_vars, reduction, reduction_op, reduction_vars, simd = match
        
        if private:
            result['private'] = {}
            result['private']['vars'] = private_vars.split()
            # result['private']['vars'] = private_vars.replace(' ','').split(',')
        elif reduction:
            result['reduction'] = {}
            result['reduction']['operator'] = reduction_op
            result['reduction']['vars'] = reduction_vars.split()
            # result['reduction']['vars'] = reduction_vars.replace(' ','').split(',')
        elif simd:
            result['simd'] = {}
            result['simd']['vars'] = []

    return result



def compare_directive(directive, preds, labels):
    '''
    label: for private ( ... ) 
    generated: for private ( ... )  reduction ( ... )
    private: TP   |   reduction: FP
    '''
    result = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'Illegal': 0}

    for pred, label in zip(preds, labels):

        if not omp_valid_pragma(pred) or not omp_valid_pragma(label):
            result['Illegal'] += 1
            # continue
        
        pred_dict, label_dict = pragma2dict(pred), pragma2dict(label)
        # if directive not in label_dict: continue

        if directive in pred_dict and directive in label_dict:
            result['TP'] += 1
        elif directive not in pred_dict and directive not in label_dict:
            result['TN'] += 1
        elif directive not in pred_dict and directive in label_dict:
            result['FN'] += 1
        elif directive in pred_dict and directive not in label_dict:
            result['FP'] += 1

    return result


def compare_vars(directive, preds, labels, operator=False):
    result = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    for pred, label in zip(preds, labels):

        pred_dict, label_dict = pragma2dict(pred), pragma2dict(label)

        if not operator:
            pred_vars = [] if directive not in pred_dict else pred_dict[directive]['vars']
            label_vars = [] if directive not in label_dict else label_dict[directive]['vars']

            total_vars = set(pred_vars + label_vars)

            for var in total_vars:
                if var in pred_vars and var in label_vars:
                    result['TP'] += 1
                elif var not in pred_vars and var not in label_vars:
                    result['TN'] += 1
                elif var not in pred_vars and var in label_vars:
                    result['FN'] += 1
                elif var in pred_vars and var not in label_vars:
                    result['FP'] += 1
        else:
            if pred_dict['reduction']['operator'] == label_dict['reduction']['operator']:
                result['TP'] += 1
            else:
                result['FN'] += 1

    return result


def omp_compute_score(conf_matrix, metric='acc'):
    if metric == 'precision':
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    elif metric == 'recall':
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN'])
    else:
        return (conf_matrix['TP'] + conf_matrix['TN']) / (conf_matrix['TP'] + conf_matrix['TN'] + conf_matrix['FP'] + conf_matrix['FN'])


def compare_vars_autoPar(preds, labels):
    result = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    for pred, label in zip(preds, labels):
        pred_vars = pred.split()[1:]
        label_vars = label.split()[1:]

        total_vars = set(pred_vars + label_vars)

        for var in total_vars:
            if var in pred_vars and var in label_vars:
                result['TP'] += 1
            elif var not in pred_vars and var not in label_vars:
                result['TN'] += 1
            elif var not in pred_vars and var in label_vars:
                result['FN'] += 1
            elif var in pred_vars and var not in label_vars:
                result['FP'] += 1

    return result