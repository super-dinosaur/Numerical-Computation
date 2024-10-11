import json
import numpy as np
import local_setting

from icecream import ic

if __name__ == '__main__':
    prompts = {
        'tar_func': lambda x: np.sin(x),
        'num_points': 5,
        'start': 0,
        'end': np.pi,
        'sampling_option': 'uniform'
    }
    with open('prompts.json', 'w') as file:
        json.dump(prompts, file, indent=4)
    ic(prompts)