import lib.__init__ as lib
import numpy as np

from icecream import ic

def f(x,c,d,e,f_):
    return c*np.sin(d*x) + e*np.cos(f_*x)
c,d,e,f_ = 3,1,4,2
tar_func = lambda x: f(x,c,d,e,f_)
start = -30
end = 30
number_points = 50

list_methods = [method for method in dir(lib.InterpolationToolkit) if method[0] != '_' and not method.endswith('debug')]
ic(list_methods)

if __name__ == '__main__':
    interpolator = lib.InterpolationToolkit(
        tar_func = tar_func,
        num_points = number_points,
        start = start,
        end = end
    )
    for method in list_methods:
        ic(method)
        func = getattr(interpolator,method)()
        lib.plot(
            func_fitting = func,
            func_target = tar_func,
            num_experimental_points = 9000,
            start = start,
            end = end,
            c=c,d=d,e=e,f=f_,n_plus_1=number_points
        )
