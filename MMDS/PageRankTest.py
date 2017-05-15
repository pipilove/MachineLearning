#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = '皮'
__mtime__ = '9/25/2015-025'
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
from math import e
import numpy as np


def pageRank(M, r, beta, epsilon, flag=False):
    it_count = 0
    N = r.size
    # print(N1)
    while (True):
        it_count += 1
        r_new = beta * np.dot(M, r) + (1 - beta) / N
        if flag and (it_count == 4 or it_count == 5):
            print('%s次迭代后:%s' % (it_count, r_new))
        # print(sum(abs(r - r_new)))
        if sum(abs(r - r_new)) < epsilon:
            break
        r = r_new
    return r, it_count


def question1():
    M = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 1, 1]])
    print(M)
    r = np.array([1, 1, 1]).T
    print(r)
    beta = 0.7
    print('.' * 50)

    r, it_count = pageRank(M, r, beta=beta, epsilon=pow(e, -6))
    print("%s\n%s次迭代" % (r * 3, it_count))


def question2():
    M = np.array([[0, 0, 1], [0.5, 0, 0], [0.5, 1, 0]])
    print(M)
    r = np.array([1 / 3, 1 / 3, 1 / 3]).T
    print(r)
    beta = 0.85
    print('.' * 50)

    r, it_count = pageRank(M, r, beta=beta, epsilon=pow(e, -6))
    print("%s\n%s次迭代" % (r, it_count))

    a, b, c = r
    epsilon = pow(e, -6)
    if abs(0.95 * c - (0.9 * b + 0.475 * a)) < epsilon:
        print('True1')
    if abs(c - (0.9 * b + 0.475 * a)) < epsilon:
        print('True2')
    if abs(0.85 * a - (c + 0.15 * b)) < epsilon:
        print('True3')
    if abs(c - (b + 0.575 * a)) < epsilon:
        print('True4')


def question3():
    M = np.array([[0, 0, 1], [0.5, 0, 0], [0.5, 1, 0]])
    print(M)
    r = np.array([1, 1, 1]).T
    print(r)
    beta = 1
    print('.' * 50)
    r, it_count = pageRank(M, r, beta=beta, epsilon=pow(e, -6), flag=True)
    print("%s\n%s次迭代" % (r, it_count))


question1()
print('*' * 50, '\n\n')
question2()
print('*' * 50, '\n\n')
question3()
