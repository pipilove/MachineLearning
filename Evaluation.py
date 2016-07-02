#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '机器学习系统评估'
__author__ = 'pika'
__mtime__ = '16-7-2'
__email__ = 'pipisorry@126.com'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓   ┏┓
            ┏━┛┻━━━┛┻━━┓
            ┃     ☃    ┃
            ┃  ┳┛   ┗┳ ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑 ┣┓
              ┃　永无BUG！┏┛
              ┗┓┓┏━━━┳┓┏┛
               ┃┫┫   ┃┫┫
               ┗┻┛   ┗┻┛
"""
import numpy as np


def evaluation(predict, actual):
    '''
    评估系统性能(两类分类和判决)：accuracy, precision, recall, F1
    '''
    # 将predict和actual数组转换成1,0数组
    predict = list(predict)
    actual = list(actual)
    labels = list(set(predict))
    if actual.count(labels[0]) <= actual.count(labels[1]):
        pos = labels[0]  # 少数样本作1
    else:
        pos = labels[1]
    predict = [1 if p == pos else 0 for p in predict]
    actual = [1 if p == pos else 0 for p in actual]

    true_pos = (np.array(predict) + np.array(actual)).tolist().count(2)
    true_neg = (np.array(predict) + np.array(actual)).tolist().count(0)

    # 计算accuracy, precision, recall, F1
    describe = {}
    describe['accuracy'] = ((true_pos + true_neg) / len(actual))
    describe['precision'] = true_pos / predict.count(1)
    describe['recall'] = true_pos / actual.count(1)
    describe['F1'] = 2 * (describe['precision'] * describe['recall']) / (describe['precision'] + describe['recall'])
    # print("accuracy: {}\nprecision: {}\nrecall: {}\nF1: {}\n".format(describe['accuracy'], describe['precision'],
    #                                                                  describe['recall'], describe['F1']))
    return describe
