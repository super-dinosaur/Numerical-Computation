import SCRIPT_CONFITG
import numpy as np
import os.path as osp

from lib.interpolation import InterpolationToolkit
from icecream import ic
from plot import ploter

if __name__ == '__main__':
    predictor = InterpolationToolkit.build(SCRIPT_CONFITG.PATH_PROMPTS)
    ploter.sub_error(
        x = predictor.x,
        gt_=predictor.tar_func,
        pred_=predictor.newton(),
        method='newton',
        m=9000
    )
