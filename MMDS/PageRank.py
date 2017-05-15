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
import linecache
import math
import numpy as np
import scipy as sp
from scipy import io


class GlobalPara():
    '''
    全局参数
    '''
    # 边文件相关
    edges_filename = 'E:\machine_learning\Machine_learning\MiningMassiveDatasets_JureLeskovec\web-Google.txt'
    line_start = 0  # edges_file从line_start开始读取
    line_end = -1

    # pagerank参数
    beta = 0.85  # 转移概率0.2


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


def LoadEdges():
    '''
    从边的txt文件中载入边
    '''
    filename = GlobalPara.edges_filename
    line_start = GlobalPara.line_start
    line_end = GlobalPara.line_end

    lines = linecache.getlines(filename)[line_start:line_end]
    edges = [tuple(i) for i in np.loadtxt(lines, int)]
    return edges


def LoadEdges1():
    edges = [(0, 1), (0, 2), (1, 2), (2, 2)]
    return edges


def sefl_algo():
    # pageRank()
    filename = GlobalPara.edges_filename
    line_start = GlobalPara.line_start
    line_end = GlobalPara.line_end
    beta = GlobalPara.beta

    lines = linecache.getlines(filename)[line_start:line_end]
    a = np.loadtxt(lines)
    print(a)


def networkx_algo():
    import networkx as nx
    beta = GlobalPara.beta
    edges = LoadEdges()
    G = nx.DiGraph(edges)
    # print(G.edges())
    pagerank_dict = nx.pagerank_scipy(G, alpha=beta)
    print(pagerank_dict[99])


if __name__ == '__main__':
    # sefl_algo()
    networkx_algo()
