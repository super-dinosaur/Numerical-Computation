import SCRIPT_CONFITG
import numpy as np
import os.path as osp
import json

from lib.Fitting import FittingToolkit
from icecream import ic
from plot import ploter
from dump_prompts import sampling_option
path_dertas = osp.join(SCRIPT_CONFITG.PATH_DATA,'perpetuate','dertas.json') 
if __name__ == '__main__':
    predictor = FittingToolkit.build(SCRIPT_CONFITG.PATH_PROMPTS)
    S,dertas = predictor.LeastSquare()
    with open(path_dertas, 'w') as file:
        json.dump(dertas, file, indent=4)
        ic(path_dertas)
    # S = predictor.LeastSquare()
    ploter.sub_error(
        x = predictor.x,
        gt_=predictor.tar_func,
        pred_=S,
        sampling_option=sampling_option,
        m=20000
    )
    ploter.show(
        x = predictor.x,
        pred_=S,
        sampling_option=sampling_option,
        gt_=predictor.tar_func
    )
