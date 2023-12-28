#!/bin/bash

python main.py      --do_test                                                           \
                    --save_dir /home/talkad/LIGHTBITS_SHARE/outputs                     \
                    --batch_size 1                                                      \
                    --model_name compcoder                                              \
                    --num_epochs 1                                                      \
                    --device cuda                                                       \
                    --tokenizer_type GPT2BPETokenizer                                   \
                    --logger test.log         
