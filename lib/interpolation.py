import numpy as np
import json

from typing import Callable
from icecream import ic
from lib.eval import gaussian_elimination, EvalTool
from scipy.interpolate import KroghInterpolator
from scipy.misc import derivative

class InterpolationToolkit():
    def __init__(self, 
                 tar_func: Callable[[float],float], 
                 num_points: int, 
                 start: float, 
                 end: float,
                 sampling_option: str = 'uniform'
    ):
        self.tar_func = tar_func
        self.num_points = num_points
        self.start = start
        self.end = end
        if sampling_option == 'uniform':
            self.x = np.linspace(start, end, num_points)
        elif sampling_option == 'Chebyshev':
            self.x = EvalTool.chebyshev_nodes(start, end, num_points-1)
        self.y = self.tar_func(self.x)

    @staticmethod   
    def build(path_prompts:str)->'InterpolationToolkit':
        with open(path_prompts, 'r') as file:
            prompts = json.load(file)
        return InterpolationToolkit(**prompts)
    
    def newton(self)->Callable[[float],float]:
        div_diff_table = EvalTool.mean_diff_table(self.x,self.y)
        def N(x):
            n = len(self.x)-1
            res = self.y[0]
            for i in range(n):
                correction = div_diff_table[i+1,i+1]*np.prod(np.array([x-self.x[j] for j in range(i+1)]))
                res += correction
            return res
        _N = np.vectorize(N)
        _N.__name__ = 'newton'
        return _N
    