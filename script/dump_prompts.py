import json
import numpy as np
import SCRIPT_CONFITG
import os.path as osp

from icecream import ic
sampling_option = 'Chebyshev'

if __name__ == '__main__':
    # modify the prompts every time you run this script
    #assert sampling option in ['uniform','Chebyshev','random']
    # prompts = {
    #     'tar_func': 'lambda x: 1/(1+25*x**2)',
    #     'num_points': 7,
    #     'start': -1,
    #     'end': 1,
    #     'sampling_option': sampling_option
    # }
    prompts = {
        'tar_func': 'lambda x: 5*np.sin(1*x) + 4*np.cos(3*x)',
        'num_points': 15,
        'start': -9,
        'end': 1,
        'sampling_option': sampling_option
    }
    path_prompts = osp.join(SCRIPT_CONFITG.PATH_DATA,'prompts','prompts.json')
    with open(path_prompts, 'w') as file:
        json.dump(prompts, file, indent=4)
    ic(path_prompts)