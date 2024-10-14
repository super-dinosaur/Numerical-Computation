import numpy as np
import matplotlib.pyplot as plt
import SCRIPT_CONFITG
from lib.Chebytool import ChebyshevApproximator
from typing import Callable

def target_function(x):
    return 5 * np.sin(x) + 4 * np.cos(3 * x)

def generate_chebyshev_nodes(start: float, end: float, num_points: int) -> np.ndarray:
    k = np.arange(1, num_points + 1)
    x_cheb = np.cos((2 * k - 1) / (2 * num_points) * np.pi)
    x_nodes = 0.5 * (x_cheb + 1) * (end - start) + start
    return x_nodes

def generate_samples(tar_func: Callable[[float], float], start: float, end: float, num_points: int, noise_std: float = 0.5):
    x_samples = generate_chebyshev_nodes(start, end, num_points)
    y_samples = tar_func(x_samples)
    y_noisy = y_samples + np.random.normal(0, noise_std, size=y_samples.shape)
    return x_samples, y_noisy

def ransac(
    x_data: np.ndarray,
    y_data: np.ndarray,
    model_func: Callable[[np.ndarray, np.ndarray], Callable[[float], float]],
    loss_func: Callable[[float, float], float],
    threshold: float,
    max_iterations: int,
    min_inliers: int = 50
):
    best_model = None
    best_inliers = []
    best_error = np.inf

    num_samples = len(x_data)

    for iteration in range(max_iterations):
        # 随机选择 5 个不重复的索引
        sample_indices = np.random.choice(num_samples, 5, replace=False)
        x_sample = x_data[sample_indices]
        y_sample = y_data[sample_indices]

        # 使用采样点拟合模型
        try:
            model = model_func(x_sample, y_sample)
        except np.linalg.LinAlgError:
            continue  # 如果矩阵不可逆，跳过本次迭代

        # 计算所有数据点的误差
        y_pred = model(x_data)
        errors = loss_func(x_data, y_data, model)

        # 识别内点（误差小于阈值的点）
        inliers = np.where(errors < threshold)[0]
        num_inliers = len(inliers)

        # 更新最佳模型
        if num_inliers > len(best_inliers):
            best_inliers = inliers
            best_model = model
            # 如果内点数量超过最小内点数，可以提前结束
            if num_inliers > min_inliers:
                break

    return best_model, best_inliers

def fit_chebyshev_model(x: np.ndarray, y: np.ndarray, degree: int = 4) -> Callable[[float], float]:
    x_min, x_max = x.min(), x.max()

    # 创建 ChebyshevApproximator 实例
    approximator = ChebyshevApproximator(
        tar_func=None,
        start=x_min,
        end=x_max,
        degree=degree
    )

    fitted_function, coeffs = approximator.fit_nodes(nodes=x, values=y)

    return fitted_function

from scipy.optimize import minimize_scalar

def loss_function_min_distance(x_data, y_data, model_func):
    losses = []
    for x0, y0 in zip(x_data, y_data):
        # 定义目标函数 h(x)
        def h(x):
            return (x - x0)**2 + (model_func(x) - y0)**2

        # 在区间 [start, end] 上最小化 h(x)
        res = minimize_scalar(h, bounds=(start, end), method='bounded')

        # 最小距离的平方根作为损失
        min_distance = np.sqrt(res.fun)
        losses.append(min_distance)

    return np.array(losses)

if __name__ == '__main__':
    # 设置参数
    start = 0
    end = 5
    num_points = 100
    noise_std = 1.0  # 噪声标准差
    degree = 9      # 多项式次数
    threshold = 2.0  # 误差阈值
    max_iterations = 1000
    min_inliers = 30  # 最小内点数

    # 生成采样点
    x_samples, y_noisy = generate_samples(target_function, start, end, num_points, noise_std)

    # 执行 RANSAC 算法
    best_model, best_inliers = ransac(
        x_data=x_samples,
        y_data=y_noisy,
        model_func=lambda x, y: fit_chebyshev_model(x, y, degree),
        loss_func=loss_function_min_distance,
        threshold=threshold,
        max_iterations=max_iterations,
        min_inliers=min_inliers
    )

    # 使用最佳模型的内点重新拟合模型
    x_inliers = x_samples[best_inliers]
    y_inliers = y_noisy[best_inliers]
    final_model = fit_chebyshev_model(x_inliers, y_inliers, degree)

    # 绘制结果
    x_plot = np.linspace(start, end, 1000)
    y_true = target_function(x_plot)
    y_pred = final_model(x_plot)

    plt.figure(figsize=(12, 6))
    plt.plot(x_plot, y_true, 'k--', label='True Function')
    plt.plot(x_plot, y_pred, 'r', label='RANSAC Fitted Function')
    plt.scatter(x_samples, y_noisy, color='gray', alpha=0.5, label='Noisy Samples')
    plt.scatter(x_inliers, y_inliers, color='green', label='Inliers')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('RANSAC with Chebyshev Least Squares Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()