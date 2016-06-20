#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '局部敏感哈希LSH'
__author__ = '皮'
__mtime__ = '6/1/2016-001'
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
import sys
from pprint import pprint


def minhash(shingles_array):
    '''
    minhash：将shingles转换成signatures
    '''
    # h = [lambda x: x % 5, lambda x: (2 * x + 1) % 5]
    h = [lambda x: (2 * x + 1) % 6, lambda x: (3 * x + 2) % 6, lambda x: (5 * x + 2) % 6]  # minhash function
    M_ic = np.full([len(h), shingles_array.shape[1]], fill_value=sys.maxsize, dtype=np.int64)  # slot初始化为无穷大
    # M_ic = np.full([len(h), shingles_array.shape[1]], fill_value=10, dtype=np.int64)

    for ri, r in enumerate(shingles_array):  # 对所有行迭代
        hi_r_value_array = [hi(ri) for hi in h]  # 每行行号的hash值，行号从0开始，有的是从1也可以
        # print("hash value for row{}:\n{}".format(ri, hi_r_value_array))
        for ci, c in enumerate(r):
            if c != 0:
                for i, hi_v in enumerate(hi_r_value_array):
                    if hi_v < M_ic[i, ci]:
                        M_ic[i, ci] = hi_v
        print("row{}:\n{}".format(ri, M_ic))
    return M_ic


def evaluate_s_curve():
    '''
    评估s曲线在不同s下，不同r\b值下的表现，可看出哪个点跳跳最大
    '''
    s = np.arange(.1, 1, .1)
    rb = [(3, 10), (6, 20), (5, 50)]
    s_curve = lambda s, r, b: 1 - (1 - s ** r) ** b
    result = np.array([[s_curve(si, r, b) for si in s] for r, b in rb])
    np.set_printoptions(precision=4, suppress=True)
    pprint(result)

    thresh = lambda r, b: (1 / b) ** (1 / r)
    thresh_value = np.array([thresh(r, b) for r, b in rb])
    print(thresh_value)


if __name__ == '__main__':
    # shingles_array = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]])
    shingles_array = np.array([[0, 1, 0, 1], [0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [1, 0, 0, 0]])
    signature_array = minhash(shingles_array)
    print(signature_array)
