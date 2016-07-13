#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'pika'
__mtime__ = '16-7-12'
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
import logging
import math
import time

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from numpy.random import RandomState
from scipy import spatial
from sklearn import decomposition
from sklearn.cluster import MiniBatchKMeans

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

n_row, n_col = 8, 1  # 小图绘图布局
n_components = n_row * n_col
ESTIMATOR_INDEX = [2]  # 要使用的估计方法gen_estimators的索引号


def gen_estimators():
    '''
    List of the different estimators, whether to center and transpose the problem, and whether the transformer uses the clustering API.
    '''
    rng = RandomState(0)
    estimators = [
        ('Eigenfaces - RandomizedPCA',
         decomposition.RandomizedPCA(n_components=n_components, whiten=True),
         True),

        ('Non-negative components - NMF tol=1e-4',
         decomposition.NMF(n_components=n_components, init='nndsvda', tol=1e-4, solver='cd'),
         False),

        ('Non-negative components - NMF tol=1e-6',
         decomposition.NMF(n_components=n_components, init='nndsvd', ),
         False),

        ('Independent components - FastICA',
         decomposition.FastICA(n_components=n_components, whiten=True),
         True),

        ('Sparse comp. - MiniBatchSparsePCA',
         decomposition.MiniBatchSparsePCA(n_components=n_components, alpha=0.8,
                                          n_iter=100, batch_size=3,
                                          random_state=rng),
         True),

        ('MiniBatchDictionaryLearning',
         decomposition.MiniBatchDictionaryLearning(n_components=15, alpha=0.1,
                                                   n_iter=50, batch_size=3,
                                                   random_state=rng),
         True),

        ('Cluster centers - MiniBatchKMeans',
         MiniBatchKMeans(n_clusters=n_components, tol=1e-3, batch_size=20,
                         max_iter=50, random_state=rng),
         True),

        ('Factor Analysis components - FA',
         decomposition.FactorAnalysis(n_components=n_components, max_iter=2),
         True),
    ]
    return estimators


def plot_gallery(title, images, n_col=n_col, n_row=n_row, image_shape=(128, 128)):
    '''
    绘制一幅大图images，其中images每行表示一个图
    :param n_col: :param n_row: 大图中小图显示布局
    :param image_shape: 小图shape大小
    '''
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))  # 每个小图都是2.×2.26
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        # vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape).T,
                   interpolation='nearest', )
        # vmin=-vmax, vmax=vmax)  # , origin="lower")
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    # plt.show()


def test(criterias, components_, name, train_time, image_shape):
    '''
    测试实验数据和标准数据的误差分析
    '''
    plot_gallery('criterias', criterias, image_shape=image_shape)

    components_[components_ > 0.1] = 1
    components_[components_ <= 0.1] = 0
    plot_gallery('%s - Train time %.1fs' % (name, train_time), components_, image_shape=image_shape)

    error = spatial.distance.euclidean(criterias.reshape(-1), components_.reshape(-1))
    print('error = ', error)


if __name__ == '__main__':
    # 加载数字数据
    dataset = sio.loadmat('digitsTest.mat')
    criterias = sio.loadmat('Digits.mat')['Ws'].T
    sort_index = np.array([7, 2, 5, 4, 6, 1, 3, 0])
    faces = dataset['Ao'].transpose()
    plot_gallery('', faces, 10, 10, (64, 64))

    # 加载飞机数据
    # dataset = sio.loadmat('PlanePartsTest.mat')
    # criterias = sio.loadmat('PlaneParts.mat')['Ws'].T
    # sort_index = np.array([2, 0, 4, 1, 3, -2, -1, -3])
    # faces = dataset['Ao'].transpose()
    # plot_gallery('', faces, 10, 10, (128, 128))

    n_samples, n_features = faces.shape  # (100, 64*64) (100, 128*128)
    image_shape = (math.sqrt(n_features), math.sqrt(n_features))

    # global centering
    faces_centered = faces - faces.mean(axis=0)
    # local centering
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    # print("Dataset consists of %d samples" % n_samples)

    # Do the estimation and plot it
    for name, estimator, center in [gen_estimators()[i] for i in ESTIMATOR_INDEX]:
        if center:
            data = faces_centered
        else:
            data = faces

        print("Extracting the top %d %s..." % (n_components, name))
        t0 = time.time()
        estimator.fit(data)
        train_time = (time.time() - t0)
        print("done in %0.3fs" % train_time)

        if hasattr(estimator, 'cluster_centers_'):
            components_ = estimator.cluster_centers_[:n_components]
        else:
            components_ = estimator.components_[:n_components]  # numpy.ndarray

        if hasattr(estimator, 'noise_variance_'):
            # plot_gallery("Pixelwise variance",
            #              estimator.noise_variance_.reshape(1, -1), n_col=1,
            #              n_row=1)
            pass
            # plot_gallery('%s - Train time %.1fs' % (name, train_time), components_)

        components_ = components_[sort_index]
        test(criterias, components_, name, train_time, image_shape)
    plt.show()
