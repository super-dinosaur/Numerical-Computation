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
    start = 0
    end = 0.5
    _D = InterpolationToolkit.DDN_debug(
        x=np.linspace(start,end,6),
        y=np.array([tar_func(x) for x in np.linspace(start,end,6)]),
        start=start,
        end=end,
        num_points=6
    )
    _D(float(0.048))
    a = 2


