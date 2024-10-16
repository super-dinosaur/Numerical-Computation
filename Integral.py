import numpy as np
import json
import math
from Eval import EvalTool
model_gt = lambda x: np.sqrt(x) * np.log(x)
model_gt = lambda x: math.exp(x)
# def integral(a,b,h,epsilon):
#     '''
#     @param a: float, lower bound
#     @param b: float, upper bound
#     @param h: float, step size
#     @param epsilon: float, error tolerance
#     '''
#     n = EvalTool.determine_nodes_by_precision(

#         epsilon
#         )

class IntegralPredictor:
    def __init__(self, a:float, b:float, epsilon:float):
        '''
        @param a: float, lower bound
        @param b: float, upper bound
        @param h: float, step size
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
        return EvalTool.determine_nodes_by_precision(
            self.a,
            self.b,
            self.epsilon
        )
    
    def CT(self):
        self.n = self.determine_nodes_by_precision()
        h = (self.b - self.a) / (self.n)
        sum_ = self.f_(self.a) + self.f_(self.b)
        for i in range(1, self.n):
            sum_ += 2 * self.f_(self.a + i * h)
        return sum_ * (h / 2)

if __name__ == '__main__':
    predictor = IntegralPredictor.build('params.json')