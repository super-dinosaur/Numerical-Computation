import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from typing import Callable, List
from icecream import ic

class ChebyshevApproximator:
    def __init__(self, tar_func: Callable[[float], float], start: float, end: float, num_points: int):
        self.tar_func = tar_func
        self.start = start
        self.end = end
        self.num_points = num_points
        
        # Transformation functions for changing interval to [-1, 1]
        self.transform = lambda x: (2 * x - (self.start + self.end)) / (self.end - self.start)
        self.inverse_transform = lambda x: 0.5 * ((self.end - self.start) * x + (self.start + self.end))
    
    @staticmethod
    def weight_function(x: float) -> float:
        return 1 / np.sqrt(1 - x**2)
    
    @staticmethod
    def integral(f_: Callable[[float], float], a: float, b: float) -> float:
        return quad(f_, a, b)[0]

    def get_chebyshev_basis(self) -> List[Callable[[float], float]]:
        # Chebyshev polynomials on [-1, 1]
        P = [lambda x: 1]  # P0(x) = 1
        if self.num_points >= 1:
            P.append(lambda x: x)  # P1(x) = x
        
        for k in range(2, self.num_points + 1):
            def P_next(x, P_k=P[-1], P_k_minus_1=P[-2], k=k):
                return 2 * x * P_k(x) - P_k_minus_1(x)
            P.append(P_next)
        return P

    def least_squares(self) -> Callable[[float], float]:
        # Get the Chebyshev basis functions
        chebyshev_basis = self.get_chebyshev_basis()
        coefficients = []
        
        for P_k in chebyshev_basis:
            # Define the integrand using the weight function and the target function
            integrand = lambda x: self.weight_function(x) * self.tar_func(self.inverse_transform(x)) * P_k(x)
            a_k = self.integral(integrand, -1, 1) / self.integral(lambda x: self.weight_function(x) * P_k(x) ** 2, -1, 1)
            coefficients.append(a_k)
        
        def S(x):
            # Calculate the approximation S(x) using the coefficients and basis
            x_transformed = self.transform(x)
            return sum(a_k * P_k(x_transformed) for a_k, P_k in zip(coefficients, chebyshev_basis))
        
        return np.vectorize(S)

if __name__ == '__main__':
    ic(np.random.normal(0, 0.1, 10))