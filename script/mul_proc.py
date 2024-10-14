import numpy as np
import matplotlib.pyplot as plt
import SCRIPT_CONFITG
import os.path as osp

from typing import Callable, List
from lib.Chebytool import ChebyshevApproximator

# 定义目标函数
def target_function(x):
    return 5 * np.sin(x) + 4 * np.cos(3 * x)

# 分割区间的函数
def split_interval(start: float, end: float, num_subintervals: int) -> List[tuple]:
    """
    将区间 [start, end] 分成 num_subintervals 个子区间，返回子区间列表。
    """
    points = np.linspace(start, end, num_subintervals + 1)
    intervals = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    return intervals

def piecewise_chebyshev_approximation(
    tar_func: Callable[[float], float],
    start: float,
    end: float,
    num_subintervals: int,
    degree: int,
    sampling_option: str = 'Chebyshev'  # 采样方式：'uniform' 或 'Chebyshev'
):
    # 分割区间
    intervals = split_interval(start, end, num_subintervals)
    approximations = []
    segment_data_list = []

    for idx, (a, b) in enumerate(intervals):
        # 根据采样方式选择采样点
        if sampling_option == 'Chebyshev':
            # 切比雪夫节点
            nodes = np.cos((2 * np.arange(1, degree + 2) - 1) / (2 * (degree + 1)) * np.pi)
            nodes = 0.5 * (nodes + 1) * (b - a) + a  # 映射到区间 [a, b]
        elif sampling_option == 'uniform':
            # 均匀采样点
            nodes = np.linspace(a, b, degree + 1)
        else:
            raise ValueError("Invalid sampling_option. Choose 'uniform' or 'Chebyshev'.")

        # 创建近似器实例
        approximator = ChebyshevApproximator(tar_func, a, b, degree)
        approx_func = approximator.least_squares()
        approximations.append((approx_func, (a, b)))

        # 计算误差
        x_test = np.linspace(a, b, 100)
        y_true = tar_func(x_test)
        y_approx = approx_func(x_test)
        error = np.max(np.abs(y_true - y_approx))

        # 保存每个分段的信息
        segment_data = {
            'a': a,
            'b': b,
            'nodes': nodes,
            'approx_func': approx_func,
            'error': error,
            'index': idx
        }
        segment_data_list.append(segment_data)

    def piecewise_func(x):
        # 定义分段函数
        x = np.array(x)
        y = np.zeros_like(x)
        for i in range(len(x)):
            xi = x[i]
            # 找到 xi 所属的子区间
            for approx_func, (a, b) in approximations:
                if a <= xi <= b:
                    y[i] = approx_func(xi)
                    break
        return y

    return piecewise_func, segment_data_list

if __name__ == '__main__':
    # 设置参数
    start = 1
    end = 5
    degrees = [1, 2, 3, 4]  # 多项式最高次数
    num_subintervals_list = [2, 4, 8]  # 子区间数量

    # 创建测试点用于绘图
    x_vals = np.linspace(start, end, 1000)
    y_true = target_function(x_vals)

    # 绘制结果
    for degree in degrees:
        for num_subintervals in num_subintervals_list:
            plt.figure(figsize=(12, 8))
            # 获取分段近似函数和分段数据
            approx_func, segment_data_list = piecewise_chebyshev_approximation(
                tar_func=target_function,
                start=start,
                end=end,
                num_subintervals=num_subintervals,
                degree=degree,
                sampling_option='Chebyshev'  # 或者 'uniform'
            )
            y_approx = approx_func(x_vals)

            # 绘制近似函数
            plt.plot(x_vals, y_approx, label=f'S(x) (max error: {max([sd["error"] for sd in segment_data_list]):.4f})')

            # 绘制每个分段的采样点，并将误差写入图例
            for segment_data in segment_data_list:
                nodes = segment_data['nodes']
                errors = segment_data['error']
                idx = segment_data['index']
                # 为每个分段选择不同的颜色
                color = plt.cm.rainbow(idx / num_subintervals)
                plt.scatter(nodes, target_function(nodes), color=color, marker='o', s=50, label=f'segment {idx+1} error: {errors:.4f}')

            # 绘制真实函数
            plt.plot(x_vals, y_true, 'k--', label='gt')
            plt.title(f'S(x) (digree {degree}, segment number {num_subintervals})')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid(True)
            path_save = osp.join(SCRIPT_CONFITG.PATH_DATA, f'piecewise_chebyshev_approximation_d{degree}_s{num_subintervals}.png')
            plt.savefig(path_save)
            plt.show()
            print(f'Saved to {path_save}')
            plt.close()