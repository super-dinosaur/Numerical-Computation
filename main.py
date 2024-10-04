import lib.__init__ as lib
import numpy as np

def f(x,c,d,e,f):
    return c*np.sin(d*x) + e*np.cos(f*x)
tar_func = lambda x: f(x,3,1,4,2)
start = 0
end = 100

if __name__ == '__main__':

    interpolator = lib.InterpolationToolkit(
        tar_func = tar_func,
        num_points = 30,
        start = start,
        end = end
    )
    lib.plot(
        func_fitting = interpolator.vandermonde(),
        func_target = tar_func,
        num_experimental_points = 9000,
        start = start,
        end = end
    )


# #写一个numpy库函数的lagrant插值函数px，真实函数还是f(x,1,2,3,4)，plot出来他们的对比图
#     p = np.poly1d(np.polyfit(interpolator.x,interpolator.y,interpolator.num_points-1))
#     lib.plot(
#         func_fitting = p,
#         func_target = lambda x: f(x,1,1,0,0),
#         num_experimental_points = 1000000,
#         start = 0,
#         end = 10
#     )
