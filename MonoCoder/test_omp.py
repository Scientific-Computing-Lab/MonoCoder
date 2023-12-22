import os
import torch
import logging
import hf_data_omp as data_omp
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from tqdm.auto import tqdm
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXForCausalLM
from prettytable import PrettyTable
from tokenizer import TokompilerTokenizer, _GPT2BPETokenizer
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup


logger = logging.getLogger()


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.data[idx]['input_ids']),
                'labels': torch.tensor(self.data[idx]['labels'])}


def tokenize(args, tokenizer, sample):
    max_size = 2048

    if not args.is_replaced:
        encodings = tokenizer(sample['full'], max_length=max_size, add_special_tokens=True, truncation=True, padding=True)

        if len(encodings['input_ids']) < max_size:
            encodings['input_ids'].append(tokenizer.eos_token_id)
    else:
        encodings = {}
        encodings['input_ids'] = tokenizer(sample['full'], max_length=max_size, add_special_tokens=True, truncation=True, padding=True)
        encodings['labels'] = encodings['input_ids'][:]

    return encodings


def concat_vars(pragma):
    unified_vars = []
    tokens = pragma.split()

    for idx, token in enumerate(tokens):
        if token.isnumeric():
            continue

        if token in ['var', 'arr', 'struct', 'arg'] and idx < len(tokens) - 1 and tokens[idx + 1].isnumeric():
            unified_vars.append(f'{token}_{tokens[idx + 1]}')
        else:
            unified_vars.append(token)

    return ' '.join(unified_vars)


def test(args):
    logger.info('start test')

    # TOKENIZER
    tokom_extended_tokens = ['parallel', 'private', 'reduction']

    if args.is_replaced:
        tokenizer = TokompilerTokenizer(vocab_path=args.vocab_file)
        tokenizer.add_tokens(tokom_extended_tokens)
        tokenizer.enable_padding(length=2048)
    else:
        tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, padding=True,
                                truncation=True, model_input_names=['input_ids'])
        tokenizer.pad_token = tokenizer.eos_token


    # DATA
    datasets = data_omp.build_omp_dataset(args)

    newd = []
    for i in range(len(datasets)):
        d = datasets[i]
        outd = d.map(lambda examples: tokenize(args, tokenizer, examples), remove_columns=['pragma', 'code', 'hash', 'full'])     
        newd.append(outd)

    traind, testd = newd

    if args.is_replaced:

        train_data = []
        for ids, labels in tqdm(zip(traind['input_ids'], traind['labels'])):
            train_data.append({'input_ids': ids, 'labels': labels})

        test_data = []
        for ids, labels in tqdm(zip(testd['input_ids'], testd['labels'])):
            test_data.append({'input_ids': ids, 'labels': labels})

        test_loader = DataLoader(dataset=CustomDataset(test_data), batch_size=1)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        test_loader = DataLoader(dataset=testd, batch_size=1, collate_fn=collator)



    # MODEL
    model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.save_dir, 'compcoder_bpe'))

    model.to(args.device)
    model.eval()


    progress_bar = tqdm(range(len(test_loader)))

    pred_table = PrettyTable()
    pred_table.field_names = ["Label", "Pred"]
    pred_table.align["Label"] = "l"
    pred_table.align["Pred"] = "l"

    post_process = lambda x: x
    if args.is_replaced:
        post_process = lambda x: concat_vars(x)
    

    for batch_idx, batch in enumerate(test_loader):
        tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'labels', 'mask', 'attention_mask']}

        input_ids = tensor_batch['input_ids']
        mask = torch.ones_like(input_ids)

        if args.is_replaced:
            mask[input_ids==tokenizer.pad_id] = 0
            mask[input_ids==tokenizer.eod] = 0
        else:
            mask[input_ids==tokenizer.eos_token_id] = 0

        outputs = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=256)
        logits = outputs[0]

        preds = torch.argmax(logits,dim=-1)
        labels = tensor_batch['input_ids'][0]

        if args.is_replaced:
            preds = preds[preds!=1]
            labels = labels[labels!=1]
        else:
            preds = preds[preds!=50256]
            labels = labels[labels!=50256]

        try:
            pred = tokenizer.decode(preds.tolist())
            label =  tokenizer.decode(labels.tolist())
        except:
            pred = ''

        pred_table.add_row([post_process(label[label.rfind('parallel'):]) if 'parallel' in label else 'None', 
                            post_process(pred[pred.rfind('parallel'):]) if 'parallel' in pred else 'None'])

        progress_bar.update(1)

    with open('results/compcoder_results.log', 'w') as f:
        f.write(str(pred_table))

