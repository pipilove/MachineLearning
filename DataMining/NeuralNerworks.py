#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '禒网络'
__author__ = '皮'
__mtime__ = '11/22/2015-022'
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
from Utility.PrintOptions import np_printoptions


class GlobalOption:
    epsilon = 10e-6
    L = 0.9  # 学习率

    input_layer_no = 8
    hidden_layer_no = 3
    out_layer_no = 1

    # W1 = np.random.rand(3, 8) * (2 * GlobalOption.epsilon) - GlobalOption.epsilon
    # 每行指向同样的下一个node
    W1_init = np.tile([[-0.1, -0.2, -0.3, -0.4, 0.1, 0.2, 0.3, 0.4]], [hidden_layer_no, 1])
    theta_hid_init = np.array([0.1, 0.2, -0.2]).reshape([-1, 1])
    theta_out_init = 0.1
    # 每行指向同样的下一个node
    W2_init = np.array([0.1, -0.1, 0.2]).reshape([out_layer_no, -1])

    # input_layer_no = 3
    # hidden_layer_no = 2
    # W1_init = np.reshape([[0.2, 0.4, -0.5, -0.3, 0.1, 0.2]], [hidden_layer_no, -1])
    # theta_hid_init = np.array([-0.4, 0.2]).reshape([-1, 1])
    # theta_out_init = 0.1
    # W2_init = np.array([-0.3, -0.2]).reshape([out_layer_no, -1])


sigmoid = lambda x: np.array([1 / (1 + 1.0 * e ** -x)]).reshape([-1, 1])


def nn(x, Tj):
    '''
    输入一个数据x，更新计算权重值
    '''
    # 输入转换成列向量
    x = np.array(x).reshape([-1, 1])
    # print('x = ', x)

    # 计算隐藏层输入输出
    I_hidden = GlobalOption.W1_init.dot(x) + GlobalOption.theta_hid_init
    # print('I_hidden = \n', I_hidden)
    O_hidden = sigmoid(I_hidden)
    # with np_printoptions(precision=3):
    #     print('O_hidden = \n', O_hidden)

    # 计算输出层输入输出
    I_out = GlobalOption.W2_init.dot(O_hidden) + GlobalOption.theta_out_init
    O_out = sigmoid(I_out)
    print('I_out = %.3f' % I_out)
    print('O_out = %.3f' % O_out)

    # 计算误差
    error_out = O_out * (1 - O_out) * (Tj - O_out)
    error_hidden = O_hidden * (1 - O_hidden) * error_out * GlobalOption.W2_init.reshape([-1, 1])
    # print('error_out = %.3f' % error_out)
    # with np_printoptions(precision=3):
    #     print('error_hidden = \n', error_hidden)

    # 更新参数
    for W, E, O in zip([GlobalOption.W1_init, GlobalOption.W2_init], [error_hidden, error_out], [x, O_hidden]):
        W += GlobalOption.L * E.dot(O.T)
        with np_printoptions(precision=3):
            print('W = \n', W)

    for theta_j, error_j in zip([GlobalOption.theta_hid_init, GlobalOption.theta_out_init], [error_hidden, error_out]):
        theta_j += GlobalOption.L * error_j
        with np_printoptions(precision=3):
            print('theta_j = \n', theta_j)


if __name__ == '__main__':
    x_new = [1, 0, 1, 0, 1, 0, 0, 0]
    # x_new = [1, 0, 1]
    y = 1
    nn(x_new, y)
