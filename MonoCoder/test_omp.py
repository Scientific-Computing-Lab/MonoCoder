import os
import torch
import logging
import pickle
import hf_data_omp as data_omp
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from prettytable import PrettyTable
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer
from transformers import DataCollatorForLanguageModeling


logger = logging.getLogger()


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.data[idx]['input_ids']),
                'labels': torch.tensor(self.data[idx]['labels'])}

def generate_pragma(pred):
    pragma = 'parallel for'
    pred = pred.split('||')[1:3]
    
    for clause in pred:
        vs = clause.split()
        if len(vs) < 2: 
            continue
        pragma += f" {vs[0]} ( {' '.join(vs[1:])} )"

    return pragma


def tokenize(args, tokenizer, sample, max_size=2048):

    if not args.is_replaced:
        encodings = tokenizer(sample['full'], max_length=max_size, add_special_tokens=True, truncation=True, padding=True)
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

    labels = testd['label']
    testd = testd.remove_columns('label')

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    test_loader = DataLoader(dataset=testd, batch_size=1, shuffle=False, collate_fn=collator)

    # MODEL
    model = GPTNeoXForCausalLM.from_pretrained('MonoCoder/MonoCoder_OMP')
    # model = GPTNeoXForCausalLM.from_pretrained(os.path.join(args.save_dir, args.model_name))


    model.to(args.device)
    model.eval()

    progress_bar = tqdm(range(len(test_loader)))

    pred_table = PrettyTable()
    pred_table.field_names = ["Label", "Pred"]
    pred_table.align["Label"] = "l"
    pred_table.align["Pred"] = "l"

    post_process = lambda x: generate_pragma(x)
    if args.is_replaced:
        post_process = lambda x: generate_pragma(concat_vars(x))
    
    for batch_idx, batch in enumerate(test_loader):
        tensor_batch = {k: v.to(args.device) for k, v in batch.items() if k in ['input_ids', 'mask', 'attention_mask']}

        input_ids = tensor_batch['input_ids']
        mask = torch.ones_like(input_ids)
        mask[input_ids==tokenizer.eos_token_id] = 0

        try:
            outputs = model.generate(input_ids=input_ids, attention_mask=mask, max_new_tokens=256)
        except:
            continue
        pred = outputs[0]
        label = labels[batch_idx]

        try:
            pred = tokenizer.decode(pred.tolist())
        except:
            pred = ''

        pred_table.add_row([post_process(label), post_process(pred[pred.rfind('parallel '):]) if 'parallel ' in pred else 'None'])

        progress_bar.update(1)

    with open('compcoder_results.log', 'w') as f:
        f.write(str(pred_table))

