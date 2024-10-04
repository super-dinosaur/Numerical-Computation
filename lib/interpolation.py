import numpy as np
from typing import Callable
from icecream import ic
from lib.eval import gaussian_elimination, EvalTool
from scipy.interpolate import KroghInterpolator

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

    def vandermonde_gt(self)->Callable[[float],float]:
        A = np.vander(self.x, increasing=True)
        coef = np.linalg.solve(A,self.y)
        def V(x):
            return np.sum(np.array([coef[i]*x**i for i in range(len(coef))]))
        return np.vectorize(V)
    
    def vandermonde(self)->Callable[[float],float]:
        A = np.array([[xi**i for i in range(self.num_points)] for xi in self.x])
        coef = EvalTool.gaussian_elimination(A,self.y)
        def V(x):
            return np.sum(np.array([coef[i]*x**i for i in range(len(coef))]))
        _V= np.vectorize(V)
        _V.__name__ = 'vandermonde'
        return _V
    
    def newton(self)->Callable[[float],float]:
        div_diff_table = EvalTool.mean_diff_table(self.x,self.y)
        def N(x):
            n = len(self.x)-1
            res = self.y[0]
            for i in range(n):
                correction = div_diff_table[i+1,i+1]*np.prod(np.array([x-self.x[j] for j in range(i+1)]))
                res += correction
            return res
        _N = np.vectorize(N)
        _N.__name__ = 'newton'
        return _N
    
    def newton_gt(self)->Callable[[float],float]:
        _N = KroghInterpolator(self.x,self.y)
        _N.__name__ = 'newton_gt'
        return _N
    
    def DDN(self)->Callable[[float],float]:
        div_diff_table = EvalTool.div_diff_table(self.x,self.y)
        def D(x):
            n = len(self.x)-1
            x0 = self.x[0]
            # h = (self.end - self.start) / self.num_points
            h = self.x[1] - self.x[0]
            t = (x - x0) / h
            res = self.y[0]
            for i in range(n):
                correction = div_diff_table[i+1,i+1]*np.prod(np.array([(t-j)/(j+1) for j in range(i+1)]))
                res += correction
            return res
        _D = np.vectorize(D)
        _D.__name__ = 'Divided Difference Newton'
        return _D
    
    @staticmethod
    def DDN_debug(x: np.ndarray, y: np.ndarray,end,start,num_points)->Callable[[float],float]:
        div_diff_table = EvalTool.div_diff_table(x,y)
        def D(u):
            n = len(x)-1
            x0 = x[0]
            h = (end - start) / num_points
            assert h - (x[1] - x[0]) < 1e-6
            t = (u - x0) / h
            res = y[0]
            for i in range(n):
                correction = div_diff_table[i+1,i+1]*np.prod(np.array([(t-j)/(j+1) for j in range(i+1)]))
                res += correction
            return res
        _D = np.vectorize(D)
        _D.__name__ = 'Divided Difference Newton'
        return _D
    
    def Piecewise_Linear(self)->Callable[[float],float]:
        eval.find_nearest(self.x,0)
        def PL(x):
            for i in range(len(self.x)-1):
                if x >= self.x[i] and x <= self.x[i+1]:
                    return (self.y[i+1]-self.y[i])/(self.x[i+1]-self.x[i])*(x-self.x[i])+self.y[i]
        _PL = np.vectorize(PL)
        _PL.__name__ = 'Piecewise Linear'
        return _PL