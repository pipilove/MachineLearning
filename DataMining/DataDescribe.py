#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'Descriptive items summarization - homework1'
__author__ = '皮'
__mtime__ = '10/29/2015-029'
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
import pandas as pd
import matplotlib.pyplot as plt


def get_df():
    age = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
    fat_percent = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2,
                   35.7]
    data = np.array([age, fat_percent]).T

    df = pd.DataFrame(data, columns=['age', 'fat_percent'])
    # print(df)
    return df


def get_describe(df):
    desc = df.describe()
    print(desc)
    desc = pd.DataFrame([df.median(), df.mean(), df.std(ddof=0)], index=['median', 'mean', 'std'])
    # print(desc.ix[['mean', 'std']])
    with pd.option_context('display.precision', 4):
        print(desc)
    return desc


def df_plot(df):
    # 同时绘制
    # plt.show(df.plot(kind='box'))

    # 分开绘制
    plt.show(df['age'].plot(kind='box'))
    plt.show(df['fat_percent'].plot(kind='box'))

    # 绘制散点图
    # plt.show(df.plot(kind='scatter', x='age', y='fat_percent'))


def min_max_nomal(df):
    '''
    min_max规格化df，如果不对函数外的df赋值，则不会改变df本身
    '''
    normal_min = 0.0
    normal_max = 1.0
    df = (df - df.min()) / (df.max() - df.min()) * (normal_max - normal_min) + normal_min
    # print(df)
    return df


if __name__ == '__main__':
    df = get_df()
    # print(df)

    # get_describe(df)

    df_plot(df)

    # df = min_max_nomal(df)
    # with pd.option_context('display.precision', 4):
    #     print(df)

    # print(df.corr())
