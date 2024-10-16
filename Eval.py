import numpy as np
import math
from typing import Callable


class EvalTool:
    @staticmethod   
    def determine_nodes_by_precision(a, b, epsilon,method='Composite Trapezoidal Rule')->int:
        #f_ = lambda x: np.sqrt(x) * np.log(x)
        #f_double_prime = lambda x: (2 - (math.log(x) / math.sqrt(x))) / (4 * x) - 1 / (2 * x**(3/2))        
        f_ = lambda x: math.exp(x)
        f_double_prime = lambda x: math.exp(x)
        if method == 'Composite Trapezoidal Rule':
            max_value = EvalTool.max(f_double_prime, a, b)
            nodes = math.sqrt((b-a)/12 * max_value / epsilon)
            return math.ceil(nodes)
        elif method == 'Romberg':
            return None

            
    @staticmethod
    def max(f:Callable[[float],float], lower:float, higher:float)->float:
        #return the maximum value of f(x) for x in [lower, higher]
        max_val = f(lower)
        for x in np.arange(lower, higher, 10e-6):
            max_val = max(max_val, f(x))
        return max_val

