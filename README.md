# Numerical-Computation
interpolation prob for week 2 TJU course

## 第2章  插值法作业
#### 说明：
1. 本课程作业提交的代码只能为.m或 .py 或.c/.c++。所有源代码均需自己独立完成，不能基于任何数值计算相关的算法库。
2. 本次作业需个人完成，提交形式“作业2_学号_姓名.zip”，文件内包含源代码（如有必要，可附一个readme），一个实验结果分析的word文件。
3. 完成时间：1周；截止时间：10月8日中午12点
4. 提交方式：电子版提交给课代表

#### 实现：
实现范德蒙德多项式插值、拉格朗日插值、牛顿插值、差分牛顿差值、分段线性插值、分段三次Hermite插值，并完成各方法之间的对比。

#### 输入：
- 插值区间[a,b]
- 参数c,d,e,f 作为标准函数
$f(x) = c \cdot \sin(dx) + e \cdot \cos(fx)$
的值
- 参数n+1作为采样点的个数，参数m作为实验点的个数。

#### 要求：
- 在区间[a,b]上均匀采集n+1个采集点，利用这n+1个采集点，分别使用范德蒙德多项式插值、拉格朗日插值、牛顿插值、分段线性、分段三次Hermite插值进行插值，求出L(x)。
- 选取m个点作为实验点，计算在这m个实验点上插值函数L(x)与目标函数f(x)的平均误差。同时对比各插值方法之间的精度差异。

#### 输出：
对比函数曲线，平均误差

#### 文件结构：
/your_project
│
├── /data
│   ├── input_data.csv
│   └── output_data.csv
│
├── /script
│   ├── preprocess_data.py
│   └── plot_results.py
│
├── /lib
│   ├── __init__.py              # 标记 lib 目录为一个 Python 包
│   ├── interpolation.py         # 各种插值方法
│   ├── vandermonde.py
│   ├── lagrange.py
│   ├── newton.py
│   └── hermite.py
│
├── config.py                    # 项目配置文件
│
├── run_experiment.sh            # 一键运行脚本
│
├── main.py                      # 主程序文件
│
└── README.md                    # 项目说明文档