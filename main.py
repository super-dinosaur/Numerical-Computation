import lib.__init__ as lib
import numpy as np


if __name__ == '__main__':
    def f(x,c,d,e,f):
        return c*np.sin(d*x) + e*np.cos(f*x)
    interpolator = lib.InterpolationToolkit(
        tar_func = lambda x: f(x,1,2,3,4),
        num_points = 10,
        start = 0,
        end = 10
    )
    interpolator.lagrange()
    lib.plot(
        func_fitteing = interpolator.lagrange(),
        func_target = lambda x: f(x,1,2,3,4),
        num_experimental_points = 100,
        start = 0,
        end = 10
    )

