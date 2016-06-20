#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Data Cleaning - noisy data:binning'
__author__ = '皮'
__mtime__ = '10/31/2015-031'
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


def smooth_by_mean(data, depth=5):
    '''
    数据不能完全等量划分时，并且这里是用平均值代替原有数值
    '''
    data.sort()

    bin_data = []
    for i in range((len(data) // depth)):
        bin_data.append([(sum(data[depth * (i):depth * (i + 1)]) / depth)] * depth)
    if len(data) / depth != 0:
        last_mean = [sum(data[len(data) - len(data) % depth:len(data)]) / (len(data) / depth)]
        bin_data.append(last_mean * (len(data) % depth))

    print(bin_data)
    return bin_data


def preprocess_data(data, depth=5):
    data.sort()
    data = np.array(data).reshape([-1, depth])
    # print(data)
    return data


def smooth_by_median(data, depth=5):
    '''
    数据能完全等量划分时
    '''
    data = np.tile(np.median(data, 1).reshape((-1, 1)), reps=depth)
    # print(data)
    return data


def smooth_by_boundary(data, depth=5):
    for bin in data:
        for i, bin_i in enumerate(bin):
            if bin.max() - bin_i > bin_i - bin.min():
                bin[i] = bin.min()
            else:
                bin[i] = bin.max()
    print(data)
    return data


if __name__ == '__main__':
    sales_data = [21, 16, 19, 24, 27, 23, 22, 21, 20, 17, 16, 20, 23, 22, 18, 24, 26, 25, 20, 26]
    sales_data = preprocess_data(sales_data)
    print(sales_data)
    # smooth_by_median(sales_data)
    smooth_by_boundary(sales_data)
