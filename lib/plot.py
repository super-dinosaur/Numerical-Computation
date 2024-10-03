import matplotlib.pyplot as plt
import numpy as np

from typing import Callable
from icecream import ic

def plot(func_fitteing:Callable[[float],float],
         func_target:Callable[[float],float],
         num_experimental_points:int,start:int,end:int
    )->None:
    x = np.linspace(start,end,num_experimental_points)
    y_fitting = func_fitteing(x)
    y_func_target = func_target(x)
    plt.plot(x,y_fitting,label='Fitting')
    plt.plot(x,y_func_target,label='Target')
    plt.legend()
    plt.grid(True)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Fitting vs Target')

    plt.show()


    
    