#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'CUR分解'
__author__ = '皮'
__mtime__ = '6/4/2016-004'
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


def cal_u():
    '''
    计算U矩阵
    '''
    np.set_printoptions(precision=4, suppress=True)
    w = np.array([[3, 3], [4, 4]])
    # w = np.array([[0, 5], [5, 0]])
    u, e, v = linalg.svd(w)
    print("u:\n{}\ne:\n{}\nv:\n{}\n".format(u, e, v))
    # print(np.allclose(w, (u.dot(np.diag(e)).dot(v))))
    e = np.diag([1 / ei if ei != 0 else 0 for ei in e])
    print(e)
    U = v.T.dot(e.dot(e.T)).dot(u.T)
    print(U)
    return U


U = cal_u()
C = np.array([[1.54, 1.54], [4.63, 4.63], [6.17, 6.17], [7.72, 7.72], [0, 0], [0, 0], [0, 0]])
R = np.array([[6.369, 6.369, 6.369, 0, 0], [6.359, 6.359, 6.359, 0, 0]])
print("C:\n{}\nU:\n{}\nR:\n{}\n".format(C, U, R))
M_ = C.dot(U).dot(R)
print(M_)
