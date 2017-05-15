#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'PCA降维'
__author__ = '皮'
__mtime__ = '6/3/2016-003'
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
import numpy as np
from scipy import linalg

np.set_printoptions(precision=4, suppress=True)
M = np.array([[1, 1], [2, 4], [3, 9], [4, 16]])

# 计算mtm的特征值和特征向量
mtm = M.T.dot(M)
print("mtm: \n{}".format(mtm))
eigvalue, eigvec = linalg.eig(mtm)
print("eigvalue: \n{}\neigvec: \n{}\n".format(eigvalue, eigvec))
# 验证特征值和对应特征向量是否计算正确
# print(mtm.dot(eigvec[:, 0]) - eigvec[:, 0] * eigvalue[0])
# print(mtm.dot(eigvec) - eigvec* eigvalue)

# 计算mmt的非0特征值对应的特征向量
eigvec_1 = M.dot(eigvec[:, 0])
eigvec_2 = M.dot(eigvec[:, 1])
print("m.dot(eigvec)\n{}".format(M.dot(eigvec)))

# 计算mmt的特征值和特征向量
mmt = M.dot(M.T)
print("\nmmt: \n{}".format(mmt))
eigvalue, eigvec = linalg.eig(mmt)
print("eigvalue: \n{}\neigvec: \n{}".format(eigvalue, eigvec))
# 特征向量应该归一化，即平方和为1
print("********", (eigvec ** 2).sum(axis=0), "********\n")

# 比较非0特征向量是否计算正确，与直接计算得到的某个特征向量成比例就对上了
print(eigvec_1 / eigvec.T[1])
print(eigvec_2 / eigvec.T[0])
# print([line / eigvec_1 for line in eigvec.T])
# print([line / eigvec_2 for line in eigvec.T])
