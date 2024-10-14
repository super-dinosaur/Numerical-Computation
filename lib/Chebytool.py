import numpy as np
import random
import matplotlib.pyplot as plt
import json
from scipy.integrate import quad
from typing import Callable, List
from icecream import ic
class ChebyshevApproximator:
    def __init__(self, tar_func: Callable[[float], float], start: float, end: float, degree: int):
        self.tar_func = tar_func
        self.start = start
        self.end = end
        self.degree = degree  # 多项式的最高次数
        
        # 转换函数，将区间 [start, end] 映射到 [-1, 1]
        self.transform = lambda x: (2 * x - (self.start + self.end)) / (self.end - self.start)
        self.inverse_transform = lambda x: 0.5 * ((self.end - self.start) * x + (self.start + self.end))
        
    @staticmethod
    def weight_function(x: float) -> float:
        return 1 / np.sqrt(1 - x**2)
    
    @staticmethod
    def integral(f_: Callable[[float], float], a: float, b: float) -> float:
        return quad(f_, a, b)[0]
    
    def get_chebyshev_basis(self) -> List[Callable[[float], float]]:
        # 切比雪夫多项式 T_k(x)
        P = [lambda x: np.ones_like(x)]  # T0(x) = 1
        if self.degree >= 1:
            P.append(lambda x: x)  # T1(x) = x
        
        for k in range(2, self.degree + 1):
            P_k_minus_1 = P[-1]
            P_k_minus_2 = P[-2]
            
            def make_Pk(P_k_minus_1, P_k_minus_2):
                return lambda x, P_k_minus_1=P_k_minus_1, P_k_minus_2=P_k_minus_2: \
                    2 * x * P_k_minus_1(x) - P_k_minus_2(x)
            
            P_k = make_Pk(P_k_minus_1, P_k_minus_2)
            P.append(P_k)
        return P
    
    def least_squares(self) -> Callable[[float], float]:
        # 获取切比雪夫基函数
        chebyshev_basis = self.get_chebyshev_basis()
        coefficients = []
        
        for P_k in chebyshev_basis:
            # 定义被积函数，使用权函数和目标函数
            integrand = lambda x, P_k=P_k: self.weight_function(x) * self.tar_func(self.inverse_transform(x)) * P_k(x)
            denominator = self.integral(lambda x, P_k=P_k: self.weight_function(x) * (P_k(x))**2, -1, 1)
            numerator = self.integral(integrand, -1, 1)
            a_k = numerator / denominator
            coefficients.append(a_k)
        
        def S(x):
            # 计算近似函数 S(x)
            x_transformed = self.transform(x)
            return sum(a_k * P_k(x_transformed) for a_k, P_k in zip(coefficients, chebyshev_basis))
        
        return np.vectorize(S)