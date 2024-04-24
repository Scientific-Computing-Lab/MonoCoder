from openai import OpenAI
from tqdm import tqdm
import json
from copy import copy
import os


prompts_dir = '/home/talkad/Downloads/ParEval-develop/prompts/raw'
serial_prompts = []
omp_prompts = []

for algo in os.listdir(prompts_dir):
    algo_dir = os.path.join(prompts_dir, algo)

    for spec in os.listdir(algo_dir):
        spec_dir = os.path.join(algo_dir, spec)

        with open(os.path.join(spec_dir, 'serial')) as f_ser, \
                open(os.path.join(spec_dir, 'omp')) as f_omp:
            serial_prompts.append({'algo': algo, 'spec':spec, 'prompt':f_ser.read()})
            omp_prompts.append({'algo': algo, 'spec':spec, 'prompt':f_omp.read()})


with open("openai.key", "r") as f:
    key = f.read()

client = OpenAI(api_key=key)

with open('omp_output.log', 'w') as out:

    for idx, sample in tqdm(enumerate(omp_prompts)):

        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": sample["prompt"]}
            ]
            )

            output = copy(sample)
            output['completion'] = response.choices[0].message.content

            out.write(json.dumps(output) + '\n')
        except Exception as e:
            print(f'failed at sample {start_idx+idx}')
