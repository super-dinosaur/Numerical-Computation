import numpy as np
from lib.eval import EvalTool
from typing import Callable, List

class ChebyshevApproximator:
    def __init__(self, tar_func: Callable[[float], float], start: float, end: float, degree: int):
        self.tar_func = tar_func
        self.start = start
        self.end = end
        self.degree = degree  # Highest degree of the polynomial

        # Transformation functions to map [start, end] to [-1, 1]
        self.transform = lambda x: (2 * x - (self.start + self.end)) / (self.end - self.start)
        self.inverse_transform = lambda x: 0.5 * ((self.end - self.start) * x + (self.start + self.end))

    @staticmethod
    def weight_function(x: float) -> float:
        return 1 / np.sqrt(1 - x**2)

    @staticmethod
    def numerical_integration(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
        # Simpson's rule
        x = np.linspace(a, b, n+1)
        h = (b - a) / n
        fx = f(x)
        integral = fx[0] + fx[-1] + 4 * np.sum(fx[1:-1:2]) + 2 * np.sum(fx[2:-2:2])
        integral *= h / 3
        return integral

    def get_chebyshev_basis(self) -> List[Callable[[float], float]]:
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

    def get_chebyshev_basis_array(self, x: np.ndarray) -> np.ndarray:
        x = self.transform(x)
        T = np.zeros((len(x), self.degree + 1))
        T[:, 0] = 1
        if self.degree >= 1:
            T[:, 1] = x
        for k in range(2, self.degree + 1):
            T[:, k] = 2 * x * T[:, k - 1] - T[:, k - 2]
        return T

    def least_squares(self) -> Callable[[float], float]:
        chebyshev_basis = self.get_chebyshev_basis()
        coefficients = []

        for P_k in chebyshev_basis:
            # Define the integrand using the weight function and target function
            integrand_num = lambda x: self.weight_function(x) * self.tar_func(self.inverse_transform(x)) * P_k(x)
            numerator = self.numerical_integration(integrand_num, -1, 1)

            integrand_den = lambda x: self.weight_function(x) * (P_k(x))**2
            denominator = self.numerical_integration(integrand_den, -1, 1)

            a_k = numerator / denominator
            coefficients.append(a_k)

        def S(x):
            x_transformed = self.transform(x)
            return sum(a_k * P_k(x_transformed) for a_k, P_k in zip(coefficients, chebyshev_basis))

        return np.vectorize(S)

    def fit_nodes(self, nodes: np.ndarray, values: np.ndarray):
        x_transformed = self.transform(nodes)
        A = np.zeros((len(nodes), self.degree + 1))
        chebyshev_basis = self.get_chebyshev_basis()
        for k, P_k in enumerate(chebyshev_basis):
            A[:, k] = P_k(x_transformed)

        # Solve (A^T A) c = A^T y using your own Gaussian elimination
        ATA = A.T @ A
        ATy = A.T @ values
        c = EvalTool.gaussian_elimination(ATA, ATy)

        def S(x):
            x_t = self.transform(x)
            return sum(c_k * P_k(x_t) for c_k, P_k in zip(c, chebyshev_basis))

        return np.vectorize(S), c