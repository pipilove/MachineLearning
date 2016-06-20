#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '贝叶斯网络结构学习-Chow-Liu算法'
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
import numpy as np
import math

TRAIN_FILE = r'./trainingData.txt.txt'
TEST_FILE = r'./testingData.txt.txt'

X = dict()
R = dict()


def f_uv(u, v, i, j, x):
    A, B = x[:, i], x[:, j]
    count = 0
    n = len(A)
    for k in range(n):
        if A[k] == u and B[k] == v:
            count += 1
    f_uv = count / n
    return f_uv


def calcu_I(x):
    m = x.shape[1]
    I = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            A, B = x[:, i], x[:, j]
            temp = 0
            for u in range(2):
                for v in range(2):
                    f_u = (A == u).sum() / len(A)
                    f_v = (B == v).sum() / len(B)
                    temp += f_uv(u, v, i, j, x) * math.log(f_uv(u, v, i, j, x) / (f_u * f_v))
            I[j][i] = I[i][j] = temp
            # print(temp)
    return I


def build_graph(I, x):
    '''
    构建图
    '''
    m = x.shape[1]
    V = range(1, m + 1)
    E = set()
    edge = []
    for i in range(m):
        for j in range(i + 1, m):
            edge.append((I[i][j], i + 1, j + 1))
            E = set(edge)
    return {'vertices': V, 'edges': E}


def make_set(v):
    X[v] = v
    R[v] = 0


def find(v):
    if X[v] != v:
        X[v] = find(X[v])
    return X[v]


def merge(v1, v2):
    r1 = find(v1)
    r2 = find(v2)
    if r1 != r2:
        if R[r1] > R[r2]:
            X[r2] = r1
        else:
            X[r1] = r2
            if R[r1] == R[r2]: R[r2] += 1


def kruskal(G):
    for vertice in graph['vertices']:
        make_set(vertice)
    max_tree = set()
    edges = list(graph['edges'])
    edges.sort(reverse=True)
    for edge in edges:
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            merge(vertice1, vertice2)
            max_tree.add(edge)
    return max_tree


if __name__ == '__main__':
    x = np.loadtxt(TRAIN_FILE, dtype=int)[:, 1:]
    graph = build_graph(calcu_I(x), x)
    result = [i[1:] for i in kruskal(graph)]
    print("Graph:{}".format(result))
