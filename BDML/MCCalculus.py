#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Monte Carlo积分计算'
__author__ = '皮'
__mtime__ = '6/15/2016-015'
__email__ = 'pipisorry@126.com'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
from random import uniform
from numpy.ma import mean, arctan, sin, var

N = 10000
f = lambda x: arctan(x) / (x ** 2 + x * sin(x))  # 要求积分的函数
a, b = 0, 1  # 积分区间
xs = [uniform(a, b) for _ in range(N)]  # 从均匀分布uniform(a,answers)生成N个样本
mean = mean([f(x) for x in xs])  # 代入积分函数，用均值去近似期望，因为函数不收敛，所以这个值也不确定
print(mean)
print(var([f(x) for x in xs]))  # 由于函数不收敛，方差巨大


def para():
    import numpy as np
    import scipy as sp
    N = 10000000
    f = lambda x: arctan(x) / (x ** 2 + x * sin(x))  # 要求积分的函数
    f = sp.vectorize(f)
    xs = np.array([random() for _ in range(N)])  # 生成N个积分区间（0，1）的数据
    fs = f(xs)
    mean = fs.mean()
    print(mean)
    var = fs.var()
    print(var)

# para()
