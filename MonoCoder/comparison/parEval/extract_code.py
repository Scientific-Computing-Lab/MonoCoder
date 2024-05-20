import json


with open('omp_output.log', 'r') as f_out, open('omp_out1.log', 'w') as f_prc:
    for line in f_out:
        sample = json.loads(line.strip())

        code = sample['completion']
        code = code[code.find('```cpp')+6:]
        code = code[:code.find('```')]

        sample['completion'] = code

        f_prc.write(json.dumps(sample) + '\n')

