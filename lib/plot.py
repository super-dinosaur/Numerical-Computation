import matplotlib.pyplot as plt
import numpy as np
import lib.LIB_CONFIG as config
import os.path as osp
import json

from typing import Callable
from icecream import ic


def plot(func_fitting:Callable[[float],float],
         func_target:Callable[[float],float],
         num_experimental_points:int,start:int,end:int,c,d,e,f,n_plus_1
    )->None:
     x = np.linspace(start,end,num_experimental_points)
     y_fitting = func_fitting(x)
     y_func_target = func_target(x)
     y_sub = np.abs(y_fitting - y_func_target)
     # 计算num_experimental_points个点的平均误差
     a = int(0.4*num_experimental_points)
     b = int(0.6*num_experimental_points)
     ic()
     # ic(x[a],x[b])
     # error_seg = np.mean(y_sub[a:b])
     error_totl = np.mean(y_sub)
     # ic(f'{error_seg:.2e}')
     ic(f'{error_totl:.2e}')

     plt.plot(x,y_sub,label='ERROR')
     plt.legend()
     plt.grid(True)
     plt.xlabel('X-axis')
     plt.ylabel('Y-axis')
     plt.title('|gt - pred|')
     #把error_total标在图上
     plt.text(x[-1],y_sub[-1],f'{error_totl:.2e}',fontsize=12)
     path_fig = osp.join(config.PATH_DATA,'fig',f'{func_fitting.__name__}.png')
     ic(path_fig)
     plt.savefig(path_fig)
     # plt.show()
     with open(osp.join(config.PATH_DATA,'error',f'{func_fitting.__name__}.json'),'w') as file:
         json.dump({
               'func_target':'%d*sin(%d*x) + %d*cos(%d*x)'%(c,d,e,f),
               'start':start,
               'end':end,
               'm':num_experimental_points,
               'n+1':n_plus_1,
               'error_totl':f'{error_totl:.2e}'
          },file,indent=4)
     plt.close()


    
    