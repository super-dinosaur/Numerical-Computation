import os.path as osp
import numpy as np
import SCRIPT_CONFITG
import pickle as pkl

from matplotlib import pyplot as plt
from icecream import ic
from typing import Callable

class ploter():
    @staticmethod
    def show(
        x:np.ndarray,
        pred_:Callable[[float],float],
        sampling_option:str,
        gt_:Callable[[float],float]=None
    ):
        x_plot = np.linspace(x[0],x[-1],20000)
        pred = pred_(x_plot)
        gt = gt_(x_plot)
        plt.plot(x_plot,pred,label=sampling_option)
        if gt_ is not None:
            plt.plot(x_plot,gt,label='Ground Truth')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title('Fitting Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        path_show = osp.join(SCRIPT_CONFITG.PATH_DATA,'show.png')
        plt.savefig(path_show)
        plt.show()
        ic(path_show)
        

    @staticmethod
    def sub_error(
        x:np.ndarray,
        gt_:Callable[[float],float],
        pred_:Callable[[float],float],
        sampling_option:str,
        m:int=9000
    ):
        x_plot = np.linspace(x[0],x[-1],m)
        gt = gt_(x_plot)
        pred = pred_(x_plot)
        error = np.abs(gt-pred)
        total_error = np.sum(error)/m
        formatted_error = f'{total_error:.2e}'
        plt.plot(x_plot,error,label=sampling_option)
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
        path_save_error = osp.join(SCRIPT_CONFITG.PATH_DATA,'error',f'{sampling_option}_error.pkl')
        dict_error = {
            'method': sampling_option,
            'error': error,
            'total_error': total_error
        }
        with open(path_save_error,'wb') as file:
            pkl.dump(dict_error,file)
        ic(path_save_error)

    @staticmethod
    def plot_contrast(path_error: dict):
        plt.figure(figsize=(10, 6))
        for i, (so, path) in enumerate(path_error.items()):
            with open(path, 'rb') as file:
                error = pkl.load(file)
            plt.plot(error['error'], label=so)
            total_error = error['total_error']
            formatted_error = f'{total_error:.2e}'
            
            # 在每条曲线的末端添加注释，位置稍微错开
            x_pos = len(error['error']) - 1
            y_pos = error['error'][-1]

            plt.annotate(f'{so} error: {formatted_error}', xy=(x_pos, y_pos), xycoords='data',
                         xytext=(5, 5 + i * 15), textcoords='offset points', ha='left', fontsize=6)
        
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title('Error Plot')
        plt.xlabel('x')
        plt.ylabel('Error')
        path_contrast = osp.join(SCRIPT_CONFITG.PATH_DATA, 'contrast_error.png')
        plt.savefig(path_contrast)
        plt.show()
        ic(path_contrast)

if __name__ == '__main__':
    path_error = {
        'uniform': osp.join(SCRIPT_CONFITG.PATH_DATA, 'error', 'uniform_error.pkl'),
        'Chebyshev': osp.join(SCRIPT_CONFITG.PATH_DATA, 'error', 'Chebyshev_error.pkl'),
        'random': osp.join(SCRIPT_CONFITG.PATH_DATA, 'error', 'random_error.pkl')
    }
    ploter.plot_contrast(path_error)