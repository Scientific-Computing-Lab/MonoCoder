#!/bin/bash

python main.py      --do_finetune                                                       \
                    --models_dir /home/talkad/shared/models/hf_checkpoints              \
                    --batch_size 2                                                      \
                    --model_name allc_gpt2tok_700M                                      \
                    --num_epochs 4                                                     \
                    --device cuda                                                       \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger bpe_monocoder.log  
