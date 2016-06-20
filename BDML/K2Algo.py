#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '贝叶斯网络结构学习-K2算法'
__author__ = 'pika'
__mtime__ = '16-5-25'
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
import math
import numpy as np
import itertools
from collections import Counter

TRAIN_FILE = r'./trainingData.txt.txt'
TEST_FILE = r'./testingData.txt.txt'


# TRAIN_FILE = r'./trainingData.txt'


def alpha_ijk(i, pi_i, x, xi_vals, ri):
    '''
    返回alpah_ijk计数矩阵ndarray，方便计算f(i, pi_i); phi：xi的parents集合pi_i值的迪卡尔积在数据集D中的出现
    '''
    index = list(pi_i)
    index.append(i)
    cnt = Counter([''.join(xi) for xi in x[:, index].astype(str)])  # 数据库D中每个迪卡尔积元素的计数
    # print(cnt)

    parent_vals = [np.unique(x[:, parent_i]) for parent_i in pi_i]  # xi的parents可能的取值list

    alpha_j = itertools.product(*parent_vals)  # xi的parents的可能取值的迪卡尔积
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is tuple else [x]
    alpha_jk = [flatten(i) for i in itertools.product(alpha_j, xi_vals)]  # xi的parents的迪卡尔积与xi可能值的迪卡尔积
    # print(alpha_jk)
    alpha_jk_cnt = np.array([cnt[''.join(i)] for i in np.array(alpha_jk).astype(str)]).reshape([-1, ri])
    # print(alpha_jk_cnt)
    return alpha_jk_cnt


def f(i, pi_i, x):
    '''
    计算给定xi的parents pi_i时数据集D的概率
    '''
    xi_vals = np.unique(x[:, i])  # xi可能的取值list
    ri = len(xi_vals)  # xi可能的取值的个数

    alpha_jk_cnt = alpha_ijk(i, pi_i, x, xi_vals, ri)
    # print(alpha_jk_cnt)

    f2 = np.array([[math.factorial(k) for k in j] for j in alpha_jk_cnt])  # f()第二项阶乘式
    f2 = [np.multiply(*k) for k in f2]
    # print(f2)

    N_ij = alpha_jk_cnt.sum(1)
    denomi = np.array([math.factorial(j) for j in (N_ij + ri - 1)])  # 第一项分母
    # print(denomi)

    return np.multiply.reduce(math.factorial(ri - 1) * f2 / denomi)  # f概率值


def k2_algorithm(x, u=1):
    '''
    :param u: parrents最大个数
    '''
    eps = 1e-30
    order = [3, 0, 1, 4, 2]

    # for i in range(len(x[0])):
    for index, i in enumerate(order):
        # Pred = set(order[:index])  # xi的前继集合（比它编号小的xi）
        Pred = set(order)
        Pred.remove(i)
        pi_i = set()  # xi的parents集合
        Pold = f(i, pi_i, x)
        print("init Pold: {}".format(Pold))

        OKToProceed = True
        while (OKToProceed and len(pi_i) < u):
            z_f = []
            for z in Pred - pi_i:
                tmp_pi_i = set()
                tmp_pi_i.update(pi_i)
                tmp_pi_i.add(z)
                z_f.append((z, f(i, tmp_pi_i, x)))
            z_f.sort(key=lambda x: x[1], reverse=True)
            Pnew = z_f[0][1] if len(z_f) > 0 else 0
            print("Pnew Node:{}, Pnew: {}".format(z_f[0][0] if len(z_f) > 0 else None, Pnew))
            if Pold - Pnew < eps and Pnew != 0:
                Pold = Pnew
                pi_i.add(z_f[0][0])
            else:
                OKToProceed = False
        print("\nNode: {}, Parents: {}".format(i, pi_i))
        print('*************************************************\n')
        # print(Pold)


if __name__ == '__main__':
    tdata = np.loadtxt(TRAIN_FILE, dtype=int)
    # tdata = np.loadtxt(TEST_FILE, dtype=int)
    x, y = tdata[:, 1:], tdata[:, 0]
    k2_algorithm(x)
