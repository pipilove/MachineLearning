#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '降维技术'
__author__ = '皮'
__mtime__ = '10/19/2015-019'
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


def PCA(X):
    # print('X = \n', X)
    dim_mean = np.mean(X, 1).reshape((-1, 1))
    X_adjust = X - dim_mean
    # print('X_adjust = \n%s\n' % X_adjust)

    cov_matrix = np.cov(X_adjust, bias=0)
    print('cov_matrix = \n', cov_matrix)

    eigenvalue, eigenvec = np.linalg.eig(cov_matrix)
    # print('eigenvalue = \n', eigenvalue)
    # print('eigenvec = \n%s\n' % eigenvec)

    sort_index = np.argsort(-eigenvalue)  # 降序排序
    eigenvec = eigenvec.T[sort_index][:1]  # 每行代表一个特征向量
    # print('eigenvec = \n', eigenvec)
    _X = eigenvec.dot(X_adjust)
    print('~X = \n%s\n' % _X)
    return _X


def SVD(X):
    autoCor = np.cov(X)
    U, E, V = linalg.svd(autoCor)
    # print('%s = \n%s * \n%s * \n%s' % (autoCor, U, np.diag(E), V))
    # U_ = U[:, 0].reshape((-1, 1)).T
    U_ = U[:, 0]
    print('~X = %s\n' % U_.dot(X))


# O_point = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6],
# [1.1, 0.9]]
O_point = [[1, 1], [2, 2], [3, 4]]
A = np.array(O_point).T
PCA(A)
SVD(A)
