import numpy as np
import json
import math
from Eval import EvalTool
from scipy.integrate import quad

# 定义被积函数
model_gt = lambda x: np.sqrt(x) * np.log(x)
# model_gt = lambda x: math.exp(x)
# model_gt = lambda x: x**(3/2)

class IntegralPredictor:
    def __init__(self, a:float, b:float, epsilon:float):
        '''
        @param a: float, lower bound
        @param b: float, upper bound
        @param epsilon: float, error tolerance
        '''
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.f_ = model_gt

    @staticmethod   
    def build(path_params):
        with open(path_params, 'r') as f:
            params = json.load(f)
        return IntegralPredictor(**params)
    
    def determine_nodes_by_precision(self):
        def f_double_prime(x):
            return (8 * x - np.log(x) - 2) / (8 * x**(3/2))
        
        # 在区间 [a, b] 上取样计算二阶导数的最大值
        max_value = f_double_prime(self.a)
        x = self.a
        while x <= self.b:
            fpp = f_double_prime(x)
            max_value = max(max_value, fpp)
            x += 10e-6

        # 修正公式，计算所需的节点数
        nodes = math.sqrt(((self.b - self.a)**3) * max_value / (12 * self.epsilon))
        return math.ceil(nodes)
    
    def CT(self):
        self.n = self.determine_nodes_by_precision()
        h = (self.b - self.a) / self.n
        sum_ = self.f_(self.a) + self.f_(self.b)
        for i in range(1, self.n):
            sum_ += 2 * self.f_(self.a + i * h)
        # Error term for the trapezoidal rule:
        # R(f) = (b-a)h^2 / 12 * f''(η), η ∈ (a, b)
        return sum_ * (h / 2), self.n, h
    
    def Romberg(self):
        self.T = np.zeros((1000, 1000))
        self.h = self.b - self.a
        self.T[0, 0] = (self.f_(self.a) + self.f_(self.b)) * self.h / 2
        for i in range(1, 1000):
            k = i  # 二分的次数
            self.h /= 2
            sum_f = 0.0
            for idx in range(1, 2 ** (k - 1) + 1):
                x = self.a + (2 * idx - 1) * self.h
                sum_f += self.f_(x)
            self.T[i, 0] = 0.5 * self.T[i - 1, 0] + self.h * sum_f
            for j in range(1, i + 1):
                self.T[i, j] = (4**j * self.T[i, j - 1] - self.T[i - 1, j - 1]) / (4**j - 1)
            if i > 1 and abs(self.T[i, i] - self.T[i - 1, i - 1]) < self.epsilon:
                return self.T[i, i], k, self.h
    
    def I_gt(self):
        gt, _ = quad(model_gt, self.a, self.b)
        return gt


if __name__ == '__main__':
    predictor = IntegralPredictor.build('params.json')
    I_CT, n, h = predictor.CT()
    print(f"Composite Trapezoidal Rule: {I_CT}, n: {n}, h: {h}")
    I_Rom, k, h = predictor.Romberg()
    print(f"Romberg: {I_Rom}, k: {k}, h: {h}")
    I_gt = predictor.I_gt() 
    print(f"Ground Truth: {I_gt}")
    print(f"Error of Composite Trapezoidal Rule: {abs(I_CT - I_gt)}")
    print(f"Error of Romberg: {abs(I_Rom - I_gt)}")
    print("Error of Composite Trapezoidal Rule is less than Romberg" if abs(I_CT - I_gt) < abs(I_Rom - I_gt) else "Error of Romberg is less than Composite Trapezoidal Rule")
    print(f"Error of Composite Trapezoidal Rule is {abs(I_CT - I_gt) / I_gt * 100:.10f}%")
    print(f"Error of Romberg is {abs(I_Rom - I_gt) / I_gt * 100:.10f}%")
