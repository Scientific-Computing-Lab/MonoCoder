# MonoCoder

With easier access to powerful compute resources, there is a growing trend in AI for software development to develop larger language models (LLMs) to address a variety of programming tasks. Even LLMs applied to tasks from the high-performance computing (HPC) domain are huge in size and demand expensive compute resources for training. This is partly because these LLMs for HPC tasks are obtained by finetuning existing LLMs that support several natural and/or programming languages. We found this design choice confusing - why do we need large LMs trained on natural languages and programming languages unrelated to HPC for HPC-specific tasks?
In this line of work, we aim to question choices made by existing LLMs by developing smaller LMs for specific domains - we call them domain-specific LMs. Specifically, we start off with HPC as a domain and build an HPC-specific LM, named MonoCoder, that is orders of magnitude smaller than existing LMs but delivers similar, if not better performance, on non-HPC and HPC tasks. Specifically, we pre-trained MonoCoder on an HPC-specific dataset (named HPCorpus) of C and C++ programs mined from GitHub. We evaluated the performance of MonoCoder against conventional multi-lingual LLMs. Results demonstrate that MonoCoder, although much smaller than existing LMs, achieves similar results on normalized-perplexity tests and much better ones in CodeBLEU competence for high-performance and parallel code generations. Furthermore, fine-tuning the base model for the specific task of parallel code generation (OpenMP parallel for pragmas) demonstrates outstanding results compared to GPT, especially when local misleading semantics are removed by our novel pre-processor Tokompiler, showcasing the ability of domain-specific models to assist in HPC-relevant tasks.

For a detailed explanation, please refer to the [full paper](https://arxiv.org/abs/2312.13322).


## Code Explanation

### Data Directory
The complete dataset can be found in the `data` directory, which also includes the train-test split for the OpenMP generation.

### MonoCoder Scripts
The `MonoCoder` directory contains two self-contained scripts that demonstrate the use of MonoCoder:

1. **`train.sh`**: This script includes the configuration for fine-tuning the model specifically for OpenMP generation.

2. **`test.sh`**: Use this script to regenerate results on the test split. It provides code for running the model on the test data.

### Hugging Face Model
The trained model is uploaded to Hugging Face and can be easily utilized in your own projects. Here's an example of how to use it in Python:

```python
from transformers import GPTNeoXForCausalLM, GPT2Tokenizer

tokenizer = GPT2Tokenizer(vocab_file=args.vocab_file, merges_file=args.merge_file, model_input_names=['input_ids'])
model = GPTNeoXForCausalLM.from_pretrained('MonoCoder/MonoCoder_OMP')
```
