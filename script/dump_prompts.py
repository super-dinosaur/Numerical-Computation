import json
import numpy as np
import SCRIPT_CONFITG
import os.path as osp

from icecream import ic
sampling_option = 'uniform'

if __name__ == '__main__':
    # modify the prompts every time you run this script
    #assert sampling option in ['uniform','Chebyshev','random']
    prompts = {
        'tar_func': 'lambda x: 1/(1+25*x**2)',
        'num_points': 50,
        'start': -1,
        'end': 1,
        'sampling_option': sampling_option
    }
    path_prompts = osp.join(SCRIPT_CONFITG.PATH_DATA,'prompts','prompts.json')
    with open(path_prompts, 'w') as file:
        json.dump(prompts, file, indent=4)
    ic(path_prompts)