#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '谱图划分'
__author__ = '皮'
__mtime__ = '10/16/2015-016'
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
import matplotlib.pyplot as plt

from Utility.Contant import eps
from Utility.PrintOptions import np_printoptions

G = {'V': [0, 1, 2, 3, 4, 5], 'E': [(0, 1), (0, 2), (1, 3), (2, 3), (1, 5), (3, 4), (4, 5)]}


# G = {'V': [0, 1, 2, 3, 4, 5], 'E': [(0, 1), (0, 2), (0, 4), (2, 3), (1, 2), (3, 4), (3, 5), (4, 5)]}


def BuildLaplacian(G):
    '''
    通过图G构造拉普拉斯矩阵L
    '''
    V = G['V']
    E = G['E']
    A = np.zeros([len(V), len(V)])
    for s_nodeid, e_nodeid in E:
        A[s_nodeid, e_nodeid] = A[e_nodeid, s_nodeid] = 1
    # print(A)
    D = np.diag(np.sum(A, 1))
    # print(D)
    L = D - A
    # print(L)
    return L


def EigenCalculate(L):
    '''
    拉普拉斯矩阵L的特征值和特征向量计算
    '''
    eigen_value, eigen_vec = linalg.eig(L)  # eigen_vec的列i对应于第i个特征值
    eigen_vec = eigen_vec.T  # 转换为eigen_vec的行i对应于第i个特征值

    # with np_printoptions(precision=1, suppress=True):
    #     print('original eigen_value = \n%s' % eigen_value.real)
    #     print('original eigen_vec = \n%s' % eigen_vec)
    sort_index = np.argsort(eigen_value)
    # print('sort_index = %s' % sort_index)

    # eigen_value = eigen_value[sort_index]
    # eigen_vec = eigen_vec[sort_index]
    # with np_printoptions(precision=1, suppress=True):
    #     print('eigen_value = \n%s' % eigen_value.real)
    #     print('eigen_vec = \n%s' % eigen_vec)

    return eigen_value[sort_index], eigen_vec[sort_index]


def PlotEigenvalue(Y, spliting_point=0, vertice_strat_index=1):
    '''
    绘制特征向量的值分布
    '''
    # plt.ion()
    x_min, x_max = vertice_strat_index, len(Y) + vertice_strat_index - 1
    y_min, y_max = min(Y), max(Y)
    margin = [1, 0.1]
    X = range(vertice_strat_index, len(Y) + vertice_strat_index)
    plt.plot(X, Y, 'r-o', label='')
    plt.axhline(spliting_point)

    # 设置图形展示效果
    def setAx(x_min, x_max, y_min, y_max, margin):
        plt.xlim(x_min - margin[0], x_max + margin[0])
        plt.ylim(y_min - margin[1], y_max + margin[1])
        plt.xlabel('Rank in x2')
        plt.ylabel('Values of x2')
        plt.legend(loc='lower right')
        plt.title('Components of x2')

    setAx(x_min, x_max, y_min, y_max, margin)

    plt.show(block=True)


if __name__ == '__main__':
    V = np.array(G['V']) + 1
    L = BuildLaplacian(G)
    print('L = \n%s' % L)
    eigen_value, eigen_vec = EigenCalculate(L)

    x2 = eigen_vec[1]
    with np_printoptions(precision=1, suppress=True):
        print('x2 = %s' % x2)
    # print(V[(x2 + eps) >= 0])
    # print(V[x2 <= eps])
    print(V[(x2 + eps) >= np.mean(x2)])
    print(V[(x2 - eps) <= np.mean(x2)])
    PlotEigenvalue(x2)
