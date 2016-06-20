#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '粒子群优化算法'
__author__ = '皮'
__mtime__ = '12/21/2015-021'
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
import random
import math
from sys import maxsize
import numpy as np


class GlobalOptions():
    particle_no = 300  # 粒子数
    particle_x_range = [-2, 5]  # 粒子取值范围
    init_speed_range = [0.3, 0.6]  # 粒子速度初值范围
    c = 2
    stop_count = 5  # 算法停止条件
    epsilon = pow(math.e, -6)


f = lambda x: x ** 3 - 5 * x ** 2 - 2 * x + 3  # 优化目标函数


def ParticleSwarm(max_flag=True):
    if max_flag:
        local_best_f = np.array([-maxsize + 1] * GlobalOptions.particle_no)
        global_best_f = -maxsize + 1
        func = max
        argindex = np.argmax
    else:
        local_best_f = np.array([maxsize] * GlobalOptions.particle_no)
        global_best_f = maxsize
        func = min
        argindex = np.argmin
    best_count = 0
    x_loc_best = None

    # 初始化速度及位置
    x_loc = np.array([random.uniform(*GlobalOptions.particle_x_range) for _ in range(GlobalOptions.particle_no)])
    v_spe = np.array([random.uniform(*GlobalOptions.init_speed_range) for _ in range(GlobalOptions.particle_no)])

    while 1:
        local_best_f = np.array([func(f(x), y2) for x, y2 in zip(x_loc, local_best_f)])
        tmp_global_best_f = global_best_f
        # print(tmp_global_best_f)
        global_best_f = func(global_best_f, func(local_best_f))
        # print(global_best_f)
        if abs(tmp_global_best_f - global_best_f) <= GlobalOptions.epsilon:
            best_count += 1
        else:
            x_loc_best = x_loc[argindex(local_best_f)]  # 最优值改变了，更新对应x
        if best_count >= GlobalOptions.stop_count:
            break

        # 更新速度及位置
        v_spe += GlobalOptions.c * (
            random.random() * (local_best_f - x_loc) + random.random() * (global_best_f - x_loc))
        x_loc = np.array([np.select([x > 5, x < -2, True], [5, -2, x]) for x in x_loc + v_spe])

    print('global_best = %.4f\nx_loc = %.4f' % (global_best_f, x_loc_best))


ParticleSwarm(False)  # 最小值及最小值点
ParticleSwarm()  # 最大值及最大值点
