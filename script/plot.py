import os.path as osp
import numpy as np
import SCRIPT_CONFITG

from matplotlib import pyplot as plt
from icecream import ic
from typing import Callable

class ploter():
    @staticmethod
    def sub_error(
        x:np.ndarray,
        gt_:Callable[[float],float],
        pred_:Callable[[float],float],
        method:str,
        m:int=9000
    ):
        x_plot = np.linspace(x[0],x[-1],m)
        gt = gt_(x_plot)
        pred = pred_(x_plot)
        error = np.abs(gt-pred)
        total_error = np.sum(error)/m
        formatted_error = f'{total_error:.2e}'
        plt.plot(x_plot,error,label=method)
        plt.legend()
        plt.annotate(f'Total error: {formatted_error}', xy=(0.5, 1), xycoords='axes fraction',
                     xytext=(0, 20), textcoords='offset points', ha='center', fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title('Error Plot')
        plt.xlabel('x')
        plt.ylabel('Error')
        path_sub_error = osp.join(SCRIPT_CONFITG.PATH_DATA,'sub_error.png')
        plt.savefig(path_sub_error)
        plt.show()
