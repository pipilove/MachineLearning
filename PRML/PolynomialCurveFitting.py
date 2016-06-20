#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '多项式曲线拟合'
__author__ = '皮'
__mtime__ = '11/8/2015-008'
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
import matplotlib.pyplot as plt
from scipy import linalg, stats

# 要拟合的函数
func = lambda x: np.sin(2 * np.pi * x)


def genPoints(p_no):
    '''
    获取要拟合的模拟数据
    '''
    x = np.random.rand(p_no)
    # x = np.linspace(0, 1, 10)
    # y要加上一个高斯分布N(0,0.01)随机偏差
    y = func(x) + stats.norm.rvs(loc=0, scale=0.1, size=10)
    return x, y


def drawCurveFitting(ax, w, x, y, order):
    '''
    绘制拟合曲线
    '''

    def drawSinCurve(ax):
        x = np.linspace(0, 1, 20)
        y = func(x)
        ax.plot(x, y, '--', alpha=0.6, label='sin curve')

    drawSinCurve(ax)

    def drawOriginData(ax, x, y):
        ax.scatter(x, y)

    drawOriginData(ax, x, y)

    def drawFittingCurve(ax, w, order):
        x = np.linspace(0, 1, 20)
        X = np.array([[xi ** i for i in range(order + 1)] for xi in x])
        y = X.dot(w)
        ax.plot(x, y, 'r', label='polynomial fitting curve')
        ax.set_ylim(-2, 2)

    drawFittingCurve(ax, w, order)

    def plotSetting(ax):
        ax.legend(loc='lower right')
        # plt.title('Polynomial Curve Fitting')
        # plt.xlabel('x')
        # plt.ylabel('y',rotation='horizontal')
        ax.set_title('Polynomial Curve Fitting')
        ax.set_xlabel('x', rotation='horizontal', lod=True)
        ax.set_ylabel('y', rotation='horizontal', lod=True)

    plotSetting(ax)

    plt.show()


def polynomialFit(x, y, order):
    X = np.array([[xi ** i for i in range(order + 1)] for xi in x])
    Y = np.array(y).reshape((-1, 1))
    # W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    W, _, _, _ = linalg.lstsq(X, Y)
    # print(W)
    return W


if __name__ == '__main__':
    order = 3  # 拟合多项式的阶数
    p_no = 10  # 拟合的数据点的个数

    ax = plt.subplot(111)
    x, y = genPoints(p_no)
    # print(x, '\n', y)

    W = polynomialFit(x, y, order=order)

    drawCurveFitting(ax, W, x, y, order=order)
