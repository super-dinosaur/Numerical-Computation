import SCRIPT_CONFITG
import numpy as np
import os.path as osp
import json
import argparse
from lib.Fitting import FittingToolkit
from icecream import ic
from plot import ploter
from dump_prompts import sampling_option
from typing import Callable

if __name__ == '__main__':

    predictor = FittingToolkit.build(SCRIPT_CONFITG.PATH_PROMPTS)
    S:Callable[[float],float]
    S = predictor.LeastSquare()
    x = np.linspace(predictor.start,predictor.end,4000)
    y = S(x)
    path_one_part_y = osp.join(SCRIPT_CONFITG.PATH_DATA,'part_y',f'{int(predictor.start)}_{int(predictor.end)}.json')
    with open(path_one_part_y,'w') as f:
        json.dump(y.tolist(),f,indent=4)
