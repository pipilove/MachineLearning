#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '接受拒绝采样'
__author__ = '皮'
__mtime__ = '4/8/2016-008'
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
from scipy import optimize
from scipy.stats import norm, uniform
import numpy as np
import matplotlib.pyplot as plt

trunc = [0, 4]  # 实际分布截断坐标点
p = [1, 1]  # 实际分布参数（均值，标准差）
q = [0, 2]  # 建议分布参数（均值，标准差）


def k(p, q, trunc):
    '''
    求k = max_x{p(x)/q(x)}
    '''
    trunc_factor = (norm.cdf(trunc[1], p[0], p[1]) - norm.cdf(trunc[0], p[0], p[1])) / p[1]
    print(trunc_factor)
    exp_k = lambda x: norm.pdf(x, p[0], p[1]) / norm.pdf(x, q[0], q[1]) / trunc_factor
    # exp_k = lambda x: norm.pdf(x, p[0], p[1]) / norm.pdf(x, q[0], q[1])#未截断的
    max_x = optimize.minimize_scalar(lambda x: -norm.pdf(x, p[0], p[1]) / norm.pdf(x, q[0], q[1]), bounds=trunc)['x']
    k = exp_k(max_x)
    print("max_x = {}\nk = {}\n".format(max_x, k))
    return k, max_x


def show_k(k, max_x, p, q, trunc):
    '''
    求出k后绘制建议分布概率密度和实际分布概率密度图，看p(x)和k*q(x)是否相切
    '''
    # x = np.linspace(norm.ppf(0.01, loc=p[0], scale=p[1]), norm.ppf(0.99, loc=p[0], scale=p[1]), N)
    x = np.linspace(trunc[0], trunc[1], 100)
    q = k * norm.pdf(x, loc=q[0], scale=q[1])  # 建议分布概率密度
    p = norm.pdf(x, loc=p[0], scale=p[1]) / (
        norm.cdf(trunc[1], p[0], p[1]) - norm.cdf(trunc[0], p[0], p[1]))  # 实际分布概率密度
    plt.plot(x, q, 'r')
    plt.plot(x, p, 'g')
    plt.axvline(max_x, color='b', label=max_x)  # 相切点
    plt.text(max_x, 0, str(round(max_x, 2)))
    plt.show()


def acc_rej_sample(k, p, q, trunc, N):
    '''
    接受拒绝采样
    :param N: 采样数
    '''
    z = norm.rvs(loc=q[0], scale=q[1], size=N)  # 从建议分布采样
    mu = uniform.rvs(size=N)  # 从均匀分布采样
    z = z[(mu <= norm.pdf(z, p[0], p[1]) / (k * norm.pdf(z, q[0], q[1])))]  # 接受-拒绝采样
    z = z[z >= trunc[0]]
    z = z[z <= trunc[1]]
    # print("sampled z = \n{}\n".format(z))
    return z


def show_z(z, p, trunc):
    '''
    采样得到采样样本z后看是否采样得到实际正态分布的近似
    '''
    # 采样分布概率密度图
    cnts, bins = np.histogram(z, bins=500, normed=True)
    bins = (bins[:-1] + bins[1:]) / 2
    plt.plot(bins, cnts, label='sampling dist')
    # plt.hist(z, bins=500, normed=True)

    # 实际分布概率密度图（截断后的）
    x = np.linspace(trunc[0], trunc[1], 100)
    trunc_factor = (norm.cdf(trunc[1], p[0], p[1]) - norm.cdf(trunc[0], p[0], p[1])) / p[1]
    plt.plot(x, norm.pdf(x, loc=p[0], scale=p[1]) / trunc_factor, 'r', label='real dist')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    k, max_x = k(p, q, trunc)
    # show_k(k, max_x, p, q, trunc)
    z = acc_rej_sample(k, p, q, trunc, N=10000000)
    show_z(z, p, trunc)
