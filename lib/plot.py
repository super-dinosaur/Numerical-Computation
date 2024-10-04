import matplotlib.pyplot as plt
import numpy as np
import lib.LIB_CONFIG as config
import os.path as osp

from typing import Callable
from icecream import ic

def plot(func_fitting:Callable[[float],float],
         func_target:Callable[[float],float],
         num_experimental_points:int,start:int,end:int
    )->None:
     x = np.linspace(start,end,num_experimental_points)
     y_fitting = func_fitting(x)
     y_func_target = func_target(x)
     y_sub = np.abs(y_fitting - y_func_target)
     plt.plot(x,y_sub,label='Sub')
     plt.legend()
     plt.grid(True)
     plt.xlabel('X-axis')
     plt.ylabel('Y-axis')
     plt.title('|gt - pred|')
     path_fig = osp.join(config.PATH_DATA,f'{func_fitting.__name__}.png')
     ic(path_fig)
     plt.savefig(path_fig)
     plt.show()


    
    