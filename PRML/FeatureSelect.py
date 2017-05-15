#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '特征选择和提取'
__author__ = '皮'
__mtime__ = '11/12/2015-012'
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
from MachineLearning.PRML.NormBayes import calLim
from matplotlib import pyplot as plt


def WithinScatterMat(X, P):
    '''
    计算多个类的类内散布矩阵
    '''

    def OneWithinScatterMat(x):
        '''
        计算某一个类x的类内散布矩阵，矩阵s_w中的元素[line,j]代表第i维和第j维的离散程度？
        '''
        x = np.array(x)
        x_mean = x.mean(axis=0)
        s_w = (x - x_mean).T.dot(x - x_mean)
        return s_w

    s_w = sum([OneWithinScatterMat(x) * p for x, p in zip(X, P)])
    return s_w


def BetweenScatterMat(X, P):
    '''
    计算多个类的类间散布矩阵
    '''
    X = np.array(X)
    m0 = sum([x.mean(axis=0) * p for x, p in zip(X, P)])
    s_b = sum(
        [(x.mean(axis=0) - m0).reshape(-1, 2).T.dot((x.mean(axis=0) - m0).reshape(-1, 2)) * p for x, p in zip(X, P)])
    return s_b


def TestScatter():
    '''
    设有如下三类模式样本集ω1，ω2和ω3，其先验概率相等，求Sw和Sb
    '''
    X1 = [[1, 0], [2, 0], [1, 1]]
    X2 = [[-1, 0], [0, 1], [-1, 1]]
    X3 = [[-1, -1], [0, -1], [0, -2]]
    X = [X1, X2, X3]
    P = [1 / 3, 1 / 3, 1 / 3]  # 各类先验概率

    # np.set_printoptions(precision=3)

    s_w = WithinScatterMat(X, P)
    print('Sw = \n', s_w)
    s_b = BetweenScatterMat(X, P)
    print('Sb = \n', s_b)


'''
**************************************************************************************
'''


def TransformX(x1, x2, p):
    '''
    计算所有数据均值，先将其均值作为新坐标轴的原点
    '''
    mean = (p[0] * np.mean(x1, 1) + p[1] * np.mean(x2, 1)).reshape([-1, 1])
    mean = np.tile(mean, [1, x1.shape[1]])
    # print(mean)
    x1 = x1 - mean
    x2 = x2 - mean
    return x1, x2


def CalR(x):
    '''
    计算自相关矩阵
    '''
    x = np.array(x)
    return x.dot(x.T) / x.shape[1]


def CalRs(x1, x2, p):
    '''
    计算两个类的自相关矩阵
    '''
    return p[0] * CalR(x1) + p[1] * CalR(x2)


def CalEigen(r):
    '''
    计算特征值和特征向量
    '''
    return linalg.eig(r)


def Draw2DPoint(x):
    '''
    绘制样本在空间中的位置，xi行向量为坐标轴不是坐标点，否则要转置，且不是增广的
    '''
    # 计算坐标极限值
    cal_lim = calLim(x)
    x_min, x_max = cal_lim.__next__()
    y_min, y_max = cal_lim.__next__()
    # print('***', x_min, x_max, y_min, y_max, '***')
    margin = 1


    # 绘制分类点和新判定点
    def drawPoints(x):
        '''
        xi行向量为坐标轴不是坐标点，否则要转置
        '''
        for xi, marker, c, l, label in zip(x, ['o', '|', 'd'], ['y', 'g', 'answers'], [1, 2, 1],
                                           ['class0', 'class1', 'class2']):
            xs, ys = zip(xi)
            plt.scatter(xs, ys, c=c, marker=marker, s=30, linewidths=l, label=label)

    drawPoints(x)

    # 设置图形展示效果
    def setAx(x_min, x_max, y_min, y_max, margin):
        plt.xlim(x_min - margin, x_max + margin)
        plt.ylim(y_min - margin, y_max + margin)
        plt.xlabel('X')
        plt.ylabel('y', rotation='horizontal')
        plt.legend(loc='lower right')
        plt.title('Plot of class0 vs. class1')

    setAx(x_min, x_max, y_min, y_max, margin)

    plt.show()


def TestKL():
    '''
    KL变换测试，输入数据行为维度
    '''
    x1 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]]).T
    x2 = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1]]).T
    p = [0.5, 0.5]
    x1, x2 = TransformX(x1, x2, p)
    r = CalRs(x1, x2, p)
    eigen_val, eigen_vec = CalEigen(r)
    for dim in [1, 2]:  # 降维到1或者2维
        phi = eigen_vec[:, np.argsort(-eigen_val)[0:dim]]
        x1_new = phi.T.dot(x1)
        x2_new = phi.T.dot(x2)
        # print(x1_new)
        # print(x2_new)
        # print()
        if dim == 2:
            print(x1_new)
            Draw2DPoint([x1_new, x2_new])
        else:
            x1_new = np.vstack([x1_new, np.zeros(x1_new.shape)])
            x2_new = np.vstack([x2_new, np.zeros(x2_new.shape)])
            Draw2DPoint([x1_new, x2_new])


'''
**************************************************************************************
'''

if __name__ == '__main__':
    # TestScatter()
    TestKL()
