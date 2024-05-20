import json
from copy import deepcopy

samples = []
sample_info = {  
            "problem_type": None,
            "language": "cpp",
            "name": None,
            "parallelism_model": None,
            "prompt": None,
            "outputs": None
        }

with open('omp_out1.log', 'r') as omp_f, open('serial_out1.log', 'r') as ser_f:
    for omp_l, ser_l in zip(omp_f, ser_f):
        omp = json.loads(omp_l)
        ser = json.loads(ser_l)

        sample = deepcopy(sample_info)
        sample['problem_type'] = ser['algo']
        sample['name'] = ser['spec']
        sample['parallelism_model'] = 'serial'
        sample['prompt'] = ser['prompt']
        sample['outputs'] = [ser['completion']]

        samples.append(sample)

        sample = deepcopy(sample_info)
        sample['problem_type'] = omp['algo']
        sample['name'] = omp['spec']
        sample['parallelism_model'] = 'omp'
        sample['prompt'] = omp['prompt']
        sample['outputs'] = [omp['completion']]

        samples.append(sample)

with open('output.json', 'w') as f:
    json.dump(samples, f, indent=4)
