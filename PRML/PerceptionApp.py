#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '感知器算法(Perception Approach)'
__author__ = '皮'
__mtime__ = '10/8/2015-008'
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

from MachineLearning.PRML.NormBayes import draw3DDiscrimInterface, draw2DDiscrimInterface


def Perc2Class(X, W, C=1):
    '''
    2类分类感知器算法
    '''
    flag = 1
    while (flag):
        flag = 0
        for Xi in X:
            if W.dot(Xi) <= 0:
                W += C * Xi
                flag = 1
    return W


def PercNClass(X, W, C=1):
    '''
    多类分类感知器算法
    '''
    class_miss = 1
    while (class_miss):
        for i, x in enumerate(X):
            class_miss = 0
            for xi in x:  # 对每类中的点迭代
                Di = W[i].dot(xi)
                for j in range(len(X)):
                    if j != i and W[j].dot(xi) >= Di:  # 用其它类的W进行判断，错则惩罚
                        W[j] -= C * xi
                        class_miss = 1
                if class_miss == 1:  # 只要有一个类分类错，则自己要奖励
                    W[i] += C * xi
    return W


def test2Class():
    '''
    2类分类感知器算法测试
    '''
    x1 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    x2 = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]])
    # x1 = np.array([[0, 0], [0, 1]])
    # x2 = np.array([[1, 0], [1, 1]])

    def preprocess(x1, x2):
        '''
        处理类模式为增广向量，并将类2*-1
        '''
        x1 = np.hstack([x1, np.ones([len(x1), 1])])
        x2 = -np.hstack([x2, np.ones([len(x2), 1])])
        return np.vstack([x1, x2])

    x = preprocess(x1, x2)
    print("X = \n%s\n" % x)
    W = Perc2Class(x, W=np.zeros([len(x[0])]), C=1)
    print("W = %s" % W)
    # print("Discrim_interface:\t%sx = 0" % W)

    x = [x1.T, x2.T]
    if len(x[0]) == 3:
        draw3DDiscrimInterface(W, x, x_new=None)
    else:
        draw2DDiscrimInterface(W, x, x_new=None)


def testNClass():
    '''
    Note: 所有类的分界面汇聚一点时，感知器多类分类算法才有用！
    '''
    test = 1
    if test == 1:
        x1 = np.array([[-1, -1]])
        x2 = np.array([[0, 0]])
        x3 = np.array([[1, 1]])
    elif test == 2:
        x1 = np.array([[0, 0]])
        x2 = np.array([[1, 1]])
        x3 = np.array([[-1, 1]])
    else:
        x1 = np.array([[0, 0], [0, 0.5]])
        x2 = np.array([[1, 1], [1, 1.5]])
        x3 = np.array([[-1, 1], [-1, 1.5]])
    print("x1 = \n%s\nx2 = \n%s\nx3 = \n%s\n" % (x1, x2, x3))

    preprocess = lambda x: np.hstack([x, np.ones([len(x), 1])])
    x1_ = preprocess(x1)
    x2_ = preprocess(x2)
    x3_ = preprocess(x3)
    # print(np.vstack([x1_, x2_, x3_]))
    x = [x1_, x2_, x3_]
    W = np.zeros([len(x), len(x[0][0])])
    W = PercNClass(x, W)
    print("W = \n%s\n" % W)

    # 绘制判别线
    W[0], W[1], W[2] = W[0] - W[1], W[0] - W[2], W[1] - W[2]
    print(W)
    draw2DDiscrimInterface(W, [x1.T, x2.T, x3.T])


if __name__ == '__main__':
    # test2Class()
    testNClass()
