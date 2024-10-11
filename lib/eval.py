import numpy as np
import os
import os.path as osp

from numpy.typing import NDArray
from numpy import floating  
from typing import List, Tuple, Any, Callable
from icecream import ic
from scipy.integrate import quad

def gaussian_elimination(A, b):
    n = len(b)
    
    # Augment the matrix A with vector b to form [A|b]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination
    for i in range(n):
        # Pivoting: Swap rows to move the largest pivot element to the diagonal
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Make the pivot element 1 and eliminate the below rows
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

class EvalTool():
    @staticmethod   
    def gaussian_elimination(A,b):
        return gaussian_elimination(A,b)
    
    @staticmethod
    def mean_diff_table(x: np.ndarray, 
                        y: np.ndarray
        ) -> np.ndarray:
        n = len(x)-1
        table = np.zeros((n+1,n+1))
        table[:,0] = y
        for j in range(1,n+1):
            for i in range(j,n+1):
                table[i,j] = (table[i,j-1] - table[i-1,j-1])/(x[i]-x[i-j])
        return table
    
    @staticmethod
    def div_diff_table(x: np.ndarray, 
                       y: np.ndarray
        ) -> np.ndarray:
        n = len(x)-1
        table = np.zeros((n+1,n+1))
        table[:,0] = y
        for j in range(1,n+1):
            for i in range(j,n+1):
                table[i,j] = table[i,j-1] - table[i-1,j-1]
        return table
    
    @staticmethod
    def find_nearest(array: np.ndarray, 
                     value: floating
        ) -> Tuple[int, int]:
        idx1 = (np.abs(array - value)).argmin()
        array[idx1] = np.inf
        idx2 = (np.abs(array - value)).argmin()
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        return idx1, idx2
    
    @staticmethod
    def gradient(x: np.ndarray, 
                 y: np.ndarray
        ) -> np.ndarray:
        n = len(x)
        grad = np.zeros(n)
        grad[0] = (y[1] - y[0])/(x[1] - x[0])
        for i in range(1,n-1):
            grad[i] = (y[i+1] - y[i-1])/(x[i+1] - x[i-1])
        grad[n-1] = (y[n-2] - y[n-3])/(x[n-2] - x[n-3])
        # grad[n] = grad[n-1] # 不知道可不可以，但是没有的话会出问题，近似的话应该差不多，反正最后一个点也龙格了，无所谓了
        return grad
    
    @staticmethod
    def compute_derivative(f: Callable[[float], float], x_val: float, dx: float = 1e-6) -> float:
        return (f(x_val + dx) - f(x_val - dx)) / (2 * dx)

    @staticmethod   
    def chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
        """生成区间 [a, b] 上的 n+1 个切比雪夫零点"""
        i = np.arange(n+1)
        x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) / (2 * (n + 1)) * np.pi)
        return x
    
    @staticmethod
    def weight_function(x: float) -> float:
        return 1 / np.sqrt(1 - x**2)

    @staticmethod
    def integral(
        f_: Callable[[float], float],
        g_: Callable[[float], float],
        a: float,
        b: float
    ) -> float:
        # Transformation to map [a, b] to [-1, 1]
        transform = lambda x: (2 * x - (a + b)) / (b - a)
        F = lambda x: EvalTool.weight_function(transform(x)) * f_(x) * g_(x)
        
        # Perform the integral on the standard interval [-1, 1]
        result, _ = quad(F, -1, 1)
        return result * (b - a) / 2  # Adjust the integral value for the scaling factor


    @staticmethod
    def get_chebyshev_basis(
        x: np.ndarray,
        start: float,
        end: float,
        num_points: int
    ) -> List[Callable[[float], float]]:
        # Transform to map [start, end] to [-1, 1]
        transform = lambda x: (2 * x - (start + end)) / (end - start)
        inverse_transform = lambda x: 0.5 * ((end - start) * x + (start + end))
        
        P = [lambda x: 1]
        alphas = [0]
        betas = []
        alpha1 = EvalTool.integral(lambda x: transform(x), P[0], -1, 1) / EvalTool.integral(P[0], P[0], -1, 1)
        alphas.append(alpha1)
        P.append(lambda x: transform(x) - alpha1)
        
        for k in range(1, num_points):
            alpha_k1 = EvalTool.integral(lambda x: transform(x) * P[k](x), P[k], -1, 1) / EvalTool.integral(P[k], P[k], -1, 1)
            alphas.append(alpha_k1)
            beta_k = EvalTool.integral(P[k], P[k], -1, 1) / EvalTool.integral(P[k - 1], P[k - 1], -1, 1)
            betas.append(beta_k)
            def P_next(x, k=k, alpha_k1=alpha_k1, beta_k=beta_k):
                return (transform(x) - alpha_k1) * P[k](x) - beta_k * P[k - 1](x)
            P.append(P_next)
        return P
    
if __name__ == "__main__":
    x = np.linspace(0,5,5)
    y = np.sin(x)
    table = EvalTool.mean_diff_table(x,y)
    x_table = np.hstack([x.reshape(-1,1),table])
    ic(osp.join(os.getcwd(),"x_table.txt"))
    with open("x_table.txt","w") as f:
        f.write(str(x_table))
        