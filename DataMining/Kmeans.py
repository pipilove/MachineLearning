#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'kmeans clustering'
__author__ = '皮'
__mtime__ = '11/23/2015-023'
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


class GlobalOptions:
    K = 3  # 聚类数目


def GetCentroids(X):
    centroids = [[4, 2, 5], [1, 1, 1], [11, 9, 2]]
    # centroids = [[10, 5, 2], [2, 3, 2], [1, 4, 6]]
    return np.array(centroids)


def Kmeans(X, K):
    X = np.array(X)
    centroids = GetCentroids(X)
    while (1):  # 这里可以设成最大迭代次数
        with np_printoptions(precision=2):
            print('centroids = \n ', centroids)
        old_centroids = centroids

        dist_mat = spatial.distance.cdist(X, centroids, metric='euclidean')
        print('dist_mat = \n', dist_mat)
        label_list = np.argmin(dist_mat, axis=1)
        print('label_list = ', label_list)

        centroids = np.array([X[label_list == cluater_i].mean(axis=0) for cluater_i in range(K)])
        print()

        if np.equal(centroids, old_centroids).all():
            # 最终得到的聚类clusterings
            print('the final clusterings = :',
                  np.array([X[label_list == cluater_i].tolist() for cluater_i in range(K)]))
            # 最终得到的centroids
            with np_printoptions(precision=2):
                print('centroids = \n ', centroids)
            break


if __name__ == '__main__':
    K = GlobalOptions.K
    X1 = [[4, 2, 5], [10, 5, 2], [5, 8, 7]]
    X2 = [[1, 1, 1], [2, 3, 2], [3, 6, 9]]
    X3 = [[11, 9, 2], [1, 4, 6], [9, 1, 7], [5, 6, 7]]
    X = X1 + X2 + X3
    # print(X)

    Kmeans(X, K)
