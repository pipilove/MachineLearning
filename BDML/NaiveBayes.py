#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '朴素贝叶斯算法(亦适用于多类分类)'
__author__ = 'pika'
__mtime__ = '16-5-23'
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

TRAIN_FILE = r'./trainingData.txt.txt'
TEST_FILE = r'./testingData.txt.txt'


def train_naive_bayes(x, y):
    '''
    训练参数：p(c){包含每个独立的p(ci)}和p(x|c){包含每个独立的p(xi|ci)}
    '''
    p_c = {}  # p(c) = {ci : p(ci)}
    p_x_cond_c = {}  # p(x|c) = {ci : [p(xi|ci)]}
    for l in np.unique(y):
        # label l下， x=1 [xi = 1]时的概率array[p(xi=1|c=l)]; 则1-array[p(xi=1|c=l)]就是array[p(xi=0|c=l)]
        p_x_cond_c[l] = x[y == l].sum(0) / (y == l).sum()
        p_c[l] = (y == l).sum() / len(y)  # p(c=l)的概率
    print("θC: {}\n".format(p_c))
    print("θA1=0|C: {}\n".format({a[0]: 1 - a[1][0] for a in p_x_cond_c.items()}))
    print("θA1=1|C: {}\n".format({a[0]: a[1][0] for a in p_x_cond_c.items()}))
    return p_c, p_x_cond_c


def predict_naive_bayes(p_c, p_x_cond_c, new_x):
    '''
    预测每个新来单个的x的label，返回一个label单值
    '''
    # new_x在类别l下的概率array
    p_l = [(l, p_c[l] * (np.multiply.reduce(p_x_cond_c[l] * new_x + (1 - p_x_cond_c[l]) * (1 - new_x)))) for l in
           p_c.keys()]
    p_l.sort(key=lambda x: x[1], reverse=True)  # new_x在类别l下的概率array按照概率大小排序
    return p_l[0][0]  # 返回概率最大对应的label


if __name__ == '__main__':
    tdata = np.loadtxt(TRAIN_FILE, dtype=int)
    x, y = tdata[:, 1:], tdata[:, 0]
    p_c, p_x_cond_c = train_naive_bayes(x, y)

    tdata = np.loadtxt(TEST_FILE, dtype=int)
    x, y = tdata[:, 1:], tdata[:, 0]
    predict = [predict_naive_bayes(p_c, p_x_cond_c, xi) for xi, yi in zip(x, y)]
    error = (y != predict).sum() / len(y)
    print("test error: {}\n".format(error))
