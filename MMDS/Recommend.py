#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '推荐系统之content-base和collaborative filtering'
__author__ = '皮'
__mtime__ = '10/18/2015-018'
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
from scipy import spatial

from Utility.PrintOptions import np_printoptions


def Nomalize(A):
    '''
    user-item规格化：对每个元素先减行和，再减列和
    '''
    row_mean = np.mean(A, 1).reshape([len(A), 1])  # 进行广播运算
    # print(row_mean)
    A -= row_mean
    # print(A)

    col_mean = np.mean(A, 0)
    # print(col_mean.shape, col_mean)
    A -= col_mean
    with np_printoptions(precision=3):
        print(A)
    return A


def CosineDist(A, scale_alpha):
    '''
    计算行向量间的cosin相似度
    '''
    A[:, -1] *= scale_alpha
    # print(A)
    cos_dist = spatial.distance.squareform(spatial.distance.pdist(A, metric='cosine'))
    with np_printoptions(precision=3):
        print('scale_alpha = %s' % scale_alpha)
        print('\tA\t\tB\t\tC')
        print(cos_dist)
        print()


if __name__ == '__main__':
    task = 2
    if task == 1:
        A = np.array([[1, 2, 3, 4, 5], [2, 3, 2, 5, 3], [5, 5, 5, 3, 2]], dtype=float)
        # print(A)
        Nomalize(A)
    else:
        for scale_alpha in [0, 0.5, 1, 2]:
            A = np.array([[1, 0, 1, 0, 1, 2], [1, 1, 0, 0, 1, 6], [0, 1, 0, 1, 0, 2]], dtype=float)
            CosineDist(A, scale_alpha=scale_alpha)
