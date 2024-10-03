import numpy as np
from typing import Callable
from icecream import ic

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
        self.y = self.tar_func(self.x)
    
    def lagrange(self)->Callable[[float],float]:
        def lk(k,x):
            xk = self.x[k]
            return np.prod([(x - xi)/(xk - xi)]for i,xi in enumerate(self.x) if i != k)
        def L(x):
            return np.sum([self.y[k]*lk(k,x)]for k in enumerate(self.x))
        return L


    
