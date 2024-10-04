import numpy as np
from typing import Callable
from icecream import ic
from lib.eval import gaussian_elimination

class InterpolationToolkit():
    def __init__(self, 
                 tar_func: Callable[[float],float], 
                 num_points: int, 
                 start: float, 
                 end: float,
                 sampling_option: str = 'uniform'
    ):
        self.tar_func = tar_func
        self.num_points = num_points
        self.start = start
        self.end = end
        if sampling_option == 'uniform':
            self.x = np.linspace(start, end, num_points)
        self.y = self.tar_func(self.x)
        
    def lagrange(self)->Callable[[float],float]:
        def lk(k,x):
            xk = self.x[k]
            return np.prod(np.array([(x - xi)/(xk-xi) for i,xi in enumerate(self.x) if i != k]))
        def L(x):
            return np.sum(np.array([yk*lk(k,x) for k,yk in enumerate(self.y)]))
        _L = np.vectorize(L)
        _L.__name__ = 'lagrange'
        return _L

    def vandermonde_true(self)->Callable[[float],float]:
        A = np.vander(self.x, increasing=True)
        coef = np.linalg.solve(A,self.y)
        def V(x):
            return np.sum(np.array([coef[i]*x**i for i in range(len(coef))]))
        return np.vectorize(V)
    
    def vandermonde(self)->Callable[[float],float]:
        A = np.array([[xi**i for i in range(self.num_points)] for xi in self.x])
        coef = gaussian_elimination(A,self.y)
        def V(x):
            return np.sum(np.array([coef[i]*x**i for i in range(len(coef))]))
        _V= np.vectorize(V)
        _V.__name__ = 'vandermonde'
        return _V
    
    def newton(self)->Callable[[float],float]:
        def divided_diff(x,y):
            if len(x) == 1:
                return y[0]
            else:
                return (divided_diff(x[1:],y[1:]) - divided_diff(x[:-1],y[:-1]))/(x[-1]-x[0])
        def N(x):
            return np.sum(np.array([divided_diff(self.x[:i+1],self.y[:i+1])*np.prod(np.array([x-xi for xi in self.x[:i]])) for i in range(self.num_points)]))
        _N = np.vectorize(N)
        _N.__name__ = 'newton'
        return _N

    
