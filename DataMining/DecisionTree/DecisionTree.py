#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '决策树分类'
__author__ = '皮'
__mtime__ = '11/22/2015-022'
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
import math

# 计算D（用每个分类的概率P计算）中元组分类需要的期望信息
from Utility.PrintOptions import np_printoptions

info = lambda P: -sum([pi * math.log(pi, 2) for pi in P])
# 计算D（用每个分类的概率P计算）中元组分类需要的期望信息
info1 = lambda p, n: -(p / (p + n) * math.log(p / (p + n), 2) + n / (p + n) * math.log(n / (p + n), 2)) if n > 0 else 0


# 计算用属性A对D划分的信息量
def InfoA(P, N):
    return sum([(pi + ni) * info1(pi, ni) / sum(P + N) for pi, ni in zip(P, N)])


# 计算用属性A对D划分的信息增益,取max的属性
P_all = 10  # 原数据中两类分别数目
N_all = 10
Gain = lambda P, N: info1(P_all, N_all) - InfoA(P, N)

# 用属性A对D划分的两类分别的数目，gender, car type, shirt size
Ps = [[6, 4], [1, 8, 1], [3, 3, 2, 2]]
Ns = [[4, 6], [3, 0, 7], [2, 4, 2, 2]]
for P, N in zip(Ps, Ns):
    gain_a = Gain(P, N)
    print("%.3f" % gain_a)
