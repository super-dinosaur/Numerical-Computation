import SCRIPT_CONFIG
import numpy as np
from typing import Callable
from icecream import ic

from lib.interpolation import InterpolationToolkit
from lib.eval import EvalTool
if __name__ == '__main__':
    def f(x,c,d,e,f):
        return c*np.sin(d*x) + e*np.cos(f*x)
    tar_func = lambda x: f(x,0,0,1,1)
    start = -0.5
    end = 0.5
    _PL = InterpolationToolkit.Piecewise_Linear_debug(
        x=np.linspace(start,end,11),
        y=np.array([tar_func(x) for x in np.linspace(start,end,11)])
    )
    _PL(float(0.048))
