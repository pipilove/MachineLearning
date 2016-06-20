#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '皮'
__mtime__ = '9/27/2015-027'
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
from math import log
import itertools

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from numpy.linalg import det, inv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def calMean(x):
    '''
    每个类各自的均值矩阵
    '''
    return [np.mean(xi, 1) for xi in x]  # 1D col vec


def calCov(x):
    '''
    计算每个类各自的协方差矩阵
    '''
    return [np.cov(xi, bias=1) for xi in x]


def discrimValue(x, p, x_new):
    '''
    计算不同类对应的判别函数的值，并通过列表返回其值
    '''
    cov = calCov(x)
    m = calMean(x)
    return [log(p[i]) - 0.5 * log(det(C)) - 0.5 * (x_new - mi).dot(inv(C)).dot(x_new - mi) for i, (mi, C) in
            enumerate(zip(m, cov))]


def discrimFun_2var(x, p, x_new):
    '''
    判定新点x_new属于哪个类
    '''
    [cov_w1, cov_w2] = calCov(x)
    # print(cov_w1, '\n', cov_w2)
    m1, m2 = calMean(x)
    # print("m1 = %s\nm2 = %s\tm.shape = %s" % (m1, m2, m2.shape))

    if ((cov_w1 - cov_w2) == 0).all():
        print('C1 = C2')
        C_I = inv(cov_w1)
        # print("C_I = \n%s" % C_I)
        # 1D col vec auto convert to 2D row vec
        print("Discrim_interface:\n%sx + %s = 0" % (
            (m1 - m2).dot(C_I), log(p[0]) - log(p[1]) - 0.5 * m1.dot(C_I).dot(m1) + 0.5 * m2.dot(C_I).dot(m2)))

        Discrim_value = log(p[0]) - log(p[1]) + (m1 - m2).dot(C_I).dot(x_new) - 0.5 * m1.dot(C_I).dot(
            m1) + 0.5 * m2.dot(C_I).dot(m2)
        print("Discrim_value is %s\n" % Discrim_value)
        class_x = 0 if Discrim_value > 0 else 1
    else:
        print('C1 != C2')
        Discrim_value = discrimValue(x, p, x_new)
        print("Discrim_value is %s\n" % Discrim_value)
        class_x = 0 if Discrim_value[0] - Discrim_value[1] > 0 else 1
    return class_x


def calLim(x):
    '''
    计算坐标极限值,其中要注意xi行向量为坐标轴不是坐标点，否则要转置
    '''
    for cor in range(len(x[0])):
        cors = list(itertools.chain.from_iterable([xi[cor] for xi in x]))
        yield min(cors), max(cors)


def draw3DDiscrimInterface(A, x, x_new=None):
    '''
    绘制3维变量的类之间的判别界面，这里xi的行是坐标点不是坐标轴，否则要转置，且是增广矩阵
    '''
    # 计算坐标极限值
    cal_lim = calLim(x)
    x_min, x_max = cal_lim.__next__()
    y_min, y_max = cal_lim.__next__()
    z_min, z_max = cal_lim.__next__()
    # print('***', x_min, x_max, y_min, y_max, z_min, z_max, '***')
    margin = 0.1

    fig = plt.figure()
    ax = Axes3D(fig)

    # ax中绘制正方体
    def drawCube(ax):
        verts = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
        poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=0.5, alpha=0.1))

    drawCube(ax)

    # ax中绘制分类点和新判定点
    def drawPoints(ax, x):
        '''
        xi行向量为坐标轴不是坐标点，否则要转置
        '''
        for xi, marker, c, label in zip(x, ['o', 'd'], ['y', 'g'], ['class0', 'class1']):
            xs, ys, zs = zip(xi)
            ax.scatter(xs, ys, zs, c=c, marker=marker, s=50, label=label)

    drawPoints(ax, x)

    if x_new is not None:
        ax.scatter(x_new[0], x_new[1], c='r', marker='*', s=50, label='new X')

    # ax中绘制分界面
    def drawInterface(ax, A, x_min, x_max, y_min, y_max, z_min, z_max, margin):
        '''
        这里A是增广矩阵
        '''
        if A[-2] != 0:
            X = np.arange(x_min - margin, x_max + margin, 0.05)
            Y = np.arange(y_min - margin, y_max + margin, 0.05)
            X, Y = np.meshgrid(X, Y)
            Z = -1 / A[-2] * (A[0] * X + A[1] * Y + A[-1])
        elif A[1] != 0:
            X = np.arange(x_min - margin, x_max + margin, 0.05)
            Z = np.arange(z_min - margin, z_max + margin, 0.05)
            X, Z = np.meshgrid(X, Z)
            Y = -1 / A[1] * (A[0] * X + A[-1])
        else:
            Y = np.arange(y_min - margin, y_max + margin, 0.05)
            Z = np.arange(z_min - margin, z_max + margin, 0.05)
            Y, Z = np.meshgrid(Y, Z)
            X = -A[-1] / A[0]
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, label='Discrimination Interface')

    drawInterface(ax, A, x_min, x_max, y_min, y_max, z_min, z_max, margin)

    # ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    # ax中设置图形展示效果
    def setAx(ax, x_min, x_max, y_min, y_max, z_min, z_max, margin):
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(loc='lower right')
        ax.set_title('Plot of class0 vs. class1')
        ax.view_init(30, 35)

    setAx(ax, x_min, x_max, y_min, y_max, z_min, z_max, margin)

    plt.show()


def draw3DDiscrimInterface1(W, x):
    '''
    这里xi的行是坐标点不是坐标轴，否则要转置，且是增广矩阵
    '''
    fig = plt.figure()
    ax = Axes3D(fig)

    # 计算坐标极限值
    def calLim(x):
        for cor in range(len(x[0])):
            cors = list(itertools.chain.from_iterable([xi.T[cor] for xi in x]))
            yield min(cors), max(cors)

    x_min, x_max = calLim(x).__next__()
    y_min, y_max = calLim(x).__next__()
    z_min, z_max = calLim(x).__next__()
    margin = 0.1


    # ax中绘制正方体
    def drawCube(ax):
        verts = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)]
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]
        poly3d = [[verts[vert_id] for vert_id in face] for face in faces]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors='w', linewidths=0.5, alpha=0.1))

    drawCube(ax)

    # ax中绘制分类点和新判定点
    def drawPoints(ax, x):
        for xi, marker, c, label in zip(x, ['o', 'd'], ['y', 'g'], ['class0', 'class1']):
            xs, ys, zs = zip(xi.T)
            ax.scatter(xs, ys, zs, c=c, marker=marker, s=50, label=label)

    drawPoints(ax, x)

    # ax中绘制分界面
    def drawInterface(ax, A, x_min, x_max, y_min, y_max, z_min, z_max, margin):
        '''
        这里A是增广矩阵
        '''
        if A[-2] != 0:
            X = np.arange(x_min - margin, x_max + margin, 0.05)
            Y = np.arange(y_min - margin, y_max + margin, 0.05)
            X, Y = np.meshgrid(X, Y)
            Z = -1 / A[-2] * (A[0] * X + A[1] * Y + A[-1])
        elif A[1] != 0:
            X = np.arange(x_min - margin, x_max + margin, 0.05)
            Z = np.arange(z_min - margin, z_max + margin, 0.05)
            X, Z = np.meshgrid(X, Z)
            Y = -1 / A[1] * (A[0] * X + A[-1])
        else:
            Y = np.arange(y_min - margin, y_max + margin, 0.05)
            Z = np.arange(z_min - margin, z_max + margin, 0.05)
            Y, Z = np.meshgrid(Y, Z)
            X = -A[-1] / A[0]
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, label='Discrimination Interface')

    drawInterface(ax, A, x_min, x_max, y_min, y_max, z_min, z_max, margin)

    # ax中设置图形展示效果
    def setAx(ax, x_min, x_max, y_min, y_max, z_min, z_max, margin):
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend(loc='lower right')
        ax.set_title('Plot of class0 vs. class1')
        ax.view_init(30, 35)

    setAx(ax, x_min, x_max, y_min, y_max, z_min, z_max, margin)

    plt.show()


def draw2DDiscrimInterface(A, x, x_new=None):
    '''
    xi行向量为坐标轴不是坐标点，否则要转置，且不是增广的
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
        for xi, marker, c, l, label in zip(x, ['o', '|', 'd'], ['y', 'g', 'b'], [1, 2, 1],
                                           ['class0', 'class1', 'class2']):
            xs, ys = zip(xi)
            plt.scatter(xs, ys, c=c, marker=marker, s=30, linewidths=l, label=label)

    drawPoints(x)

    if x_new is not None:
        plt.scatter(x_new[0], x_new[1], c='r', marker='*', s=50, label='new X')

    # 绘制分界面
    def drawInterface(A, x_min, x_max, y_min, y_max, margin):
        if A[-2] != 0:
            X = np.arange(x_min - margin, x_max + margin, 0.05)
            Y = -1 / A[-2] * (A[0] * X + A[-1])
            plt.plot(X, Y, label='Discrimination Interface')
        else:
            plt.vlines(-A[-1] / A[0], y_min - margin, y_max + margin, label='Discrimination Interface')

    if len(A.shape) == 1:
        drawInterface(A, x_min, x_max, y_min, y_max, margin)
    else:
        for a in A:
            drawInterface(a, x_min, x_max, y_min, y_max, margin)


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


def DrawDiscrimInterface(x, p, x_new):
    '''
    计算相关值并绘制变量的类之间的判别界面
    '''
    [cov_w1, cov_w2] = calCov(x)
    m1, m2 = calMean(x)

    if ((cov_w1 - cov_w2) == 0).all():
        C_I = inv(cov_w1)
        S = log(p[0]) - log(p[1]) - 0.5 * m1.dot(C_I).dot(m1) + 0.5 * m2.dot(C_I).dot(m2)
        A = (m1 - m2).dot(C_I)
        A = np.hstack([A, S])
        # print("A = \n%s\n" % A)

        if len(x[0]) == 3:
            draw3DDiscrimInterface(A, x, x_new)
        else:
            draw2DDiscrimInterface(A, x, x_new)


if __name__ == '__main__':
    test = 1
    if test == 1:
        # 训练数据观测值
        x1 = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0]]).T  # convert to R.V. in line
        x2 = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]]).T
        # 要判别的观测值
        x_new = np.array([0.6, 1, 0])  # 1D col vec
        # x_new = np.array([0.4, 1, 0])
    elif test == 2:
        x1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]]).T
        x2 = np.array([[4, 4], [6, 4], [6, 6], [4, 6]]).T
        x_new = np.array([0.6, 1])
    else:
        x1 = np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0], [1, 1, 0]]).T
        x2 = np.array([[0, 0, 1], [0, 0.5, 1], [0, 0.5, 0], [1, 1, 1]]).T
        x_new = np.array([0.5, 1, 0])
    print("x1 = \n%s\nx2 = \n%s\n" % (x1, x2))
    # print("x_new = %s\tx_new.shape = %s" % (x_new, x_new.shape))

    # 每个类的概率
    p = [0.5, 0.5]

    x = [x1, x2]
    print("The class label of %s is %s" % (x_new, discrimFun_2var(x, p, x_new)))
    DrawDiscrimInterface(x, p, x_new)
