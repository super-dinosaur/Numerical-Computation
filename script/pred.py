import SCRIPT_CONFITG
import numpy as np
import os.path as osp

from lib.interpolation import InterpolationToolkit
from icecream import ic
from plot import ploter
from dump_prompts import sampling_option

if __name__ == '__main__':
    predictor = InterpolationToolkit.build(SCRIPT_CONFITG.PATH_PROMPTS)
    ploter.sub_error(
        x = predictor.x,
        gt_=predictor.tar_func,
        pred_=predictor.newton(),
        sampling_option=sampling_option,
        m=9000
    )
