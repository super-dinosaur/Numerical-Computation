import numpy as np
import json
import local_setting
import os.path as osp

from typing import Callable
from icecream import ic
from lib.eval import  EvalTool
from lib.Chebytool import ChebyshevApproximator

class FittingToolkit():
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
            ic()
            self.x = EvalTool.chebyshev_nodes(start, end, num_points-1)
        elif sampling_option == 'random':
            self.x = np.random.uniform(start, end, num_points)
        self.record_x(sampling_option)
        self.y = self.tar_func(self.x)

    @staticmethod   
    def build(path_prompts:str)->'FittingToolkit':
        with open(path_prompts, 'r') as file:
            prompts = json.load(file)
        prompts['tar_func'] = eval(prompts['tar_func'])
        ic(prompts['tar_func'])
        return FittingToolkit(**prompts)

    def LeastSquare(self) -> Callable[[float], float]:
        # 使用 Chebyshev 节点来拟合函数
        ch = ChebyshevApproximator(
            tar_func=self.tar_func,
            start=self.start,
            end=self.end,
            num_points=self.num_points
        )

        return ch.least_squares()
        # return ch.least_squares_perpetuated()

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
    


    def record_x(self,sampling_option:str):
        path_x = osp.join(local_setting.PATH_DATA,'prompts','x_table.json')
        dict_record = {
            'method': sampling_option,
            'x': self.x.tolist(),
        }
        with open(path_x, 'w') as file:
            json.dump(dict_record, file, indent=4)
        ic(path_x)        