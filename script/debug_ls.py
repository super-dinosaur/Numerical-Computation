import numpy as np
import SCRIPT_CONFITG
import numpy as np
import os.path as osp

from scipy.integrate import quad
from typing import List, Tuple, Any, Callable
from lib.Fitting import FittingToolkit
from icecream import ic
from plot import ploter
from dump_prompts import sampling_option
from lib.eval import EvalTool
from scipy.integrate import quad


if __name__ == '__main__':
    predictor = FittingToolkit.build(SCRIPT_CONFITG.PATH_PROMPTS)
    S_ = predictor.LeastSquare()
    
