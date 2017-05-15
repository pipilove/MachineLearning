#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '势函数法非线性分类'
__author__ = '皮'
__mtime__ = '10/23/2015-023'
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
from sympy import symbols, simplify, exp


def GetHermite(order):
    '''
    获得第一类势函数算法-Hermite多项式 势函数变量表达式
    '''
    # 选择合适的正交函数集
    x1 = symbols('x1')
    H_polynomial = [x1 ** 0, 2 * x1, 4 * x1 ** 2 - 2, 8 * x1 ** 3 - 12 * x1,
                    16 * x1 ** 4 - 48 * x1 ** 2 + 12]  # hermite多项式(前4个)
    Hx1 = H_polynomial[0:order + 1]

    # 建立二维的正交函数集
    Hx2 = [h.subs(x1, 'x2') for h in Hx1]
    phi_x = [simplify(i * j) for i in Hx1 for j in Hx2]
    # print(phi_x)

    # 生成势函数
    phi_xk = [phi_xi.subs({x1: 'xk1', 'x2': 'xk2'}) for phi_xi in phi_x]
    K_x_xk = simplify(sum([phi_xi * phi_xki for phi_xi, phi_xki in zip(phi_x, phi_xk)]))
    # print('k(x, xk) = %s\n' % K_x_xk)
    return K_x_xk


def GetExponent():
    '''
    获得第一类势函数算法-指数型势函数 势函数变量表达式
    '''
    # 生成势函数
    x1, x2 = symbols('x1 x2')
    xk1, xk2 = symbols('xk1 xk2')
    k_x_xk = simplify(exp(-1 * ((x1 - xk1) ** 2 + (x2 - xk2) ** 2)))
    # print(k_x_xk)
    return k_x_xk


def CalPotential(X, type=1, order=1):
    '''
    计算累积势函数
    '''
    if type == 1:
        K_x_xk = GetHermite(order)
    else:
        K_x_xk = GetExponent()
    print('k(x, xk) = %s\n' % K_x_xk)

    K = symbols('0')
    step = 1
    flag = 1
    while (flag):
        flag = 0
        for x in X[0]:
            if K.subs({'x1': x[0], 'x2': x[1]}) <= symbols('0'):
                K = simplify(K + K_x_xk.subs({'xk1': x[0], 'xk2': x[1]}))
                flag = 1
                # print('step %d\n%s' % (step, k))
            step += 1
        for x in X[1]:
            if K.subs({'x1': x[0], 'x2': x[1]}) >= symbols('0'):
                K = simplify(K - K_x_xk.subs({'xk1': x[0], 'xk2': x[1]}))
                flag = 1
                # print('step %d\n%s' % (step, k))
            step += 1
    # print('k(x) = %s\n' % k)
    return simplify(K)


def testForHermite():
    '''
    第一类势函数算法测试
    '''
    test = 2
    if test == 1:
        X1 = [[1, 0], [0, -1]]  # 类1
        X2 = [[-1, 0], [0, 1]]  # 类2
        order = 1
    else:
        X1 = [[0, 1], [0, -1]]
        X2 = [[1, 0], [-1, 0]]
        order = 2
    X = (X1, X2)
    K = CalPotential(X, type=1, order=order)
    print('d(x) = k(x) = %s\n' % K)


def testForExpond():
    '''
    第一类势函数算法测试
    '''
    test = 2
    if test == 1:
        X1 = [[0, 0], [2, 0]]  # 类1
        X2 = [[1, 1], [1, -1]]  # 类2
    else:
        X1 = [[0, 1], [0, -1]]  # 类1
        X2 = [[1, 0], [-1, 0]]  # 类2
    X = (X1, X2)
    K = CalPotential(X, type=2)
    print('d(x) = k(x) = %s\n' % K)


if __name__ == '__main__':
    testForHermite()
    testForExpond()
