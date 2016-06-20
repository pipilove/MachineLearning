#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = 'boosting & adboost algorithm'
__author__ = 'pika'
__mtime__ = '16-4-7'
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
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def loadSimpData():
    data_arr = np.array(
        [[1.2, 2.3], [0.9, 2.2], [1.2, 1.2], [1.1, 1.2], [1., 2.1], [1.1, 2.1], [2., 1.1], [1.3, 1.], [1., 1.],
         [2., 1.3]])
    class_labels = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1])
    return data_arr, class_labels


def stump_classifier(data_arr, dim, thresh_val, thresh_ineq):
    '''
    single stump tree classify
    '''
    pred_lab_arr = np.ones(data_arr.shape[0], dtype=int)
    if thresh_ineq == 'lt':
        pred_lab_arr[data_arr[:, dim] <= thresh_val] = 0
    else:
        pred_lab_arr[data_arr[:, dim] > thresh_val] = 0
    return pred_lab_arr


def best_weak_classifier(data_arr, class_labels, w):
    '''
    train and select lowest error weak classifier
    '''
    n, feature_n = data_arr.shape  # 样本数和维度
    best_stump = {}
    min_error = sys.maxsize
    for f in range(feature_n):  # 对每个feature迭代训练一个单桩树弱分类器
        for thresh_val in np.linspace(data_arr[:, f].min(), data_arr[:, f].max(),
                                      num=10):  # loop over all range in current dimension
            for thresh_ineq in ['lt', 'gt']:  # go over less than and greater than
                pred_lab_arr = stump_classifier(data_arr, f, thresh_val, thresh_ineq)
                error = w.T.dot((class_labels - pred_lab_arr).__abs__())
                if error < min_error:
                    # print(error)
                    min_error = error
                    best_stump['feature'] = f  # 记录弱分类器相关信息
                    best_stump['thresh_val'] = thresh_val
                    best_stump['thresh_ineq'] = thresh_ineq
                    best_stump['error'] = error
                    best_stump['error_flag'] = np.array([int(i) for i in pred_lab_arr == class_labels])
    return best_stump


def adaboost_sintree(data_arr, class_labels, num_it=40):
    '''
    基于单层决策树的AdaBoost
    '''
    num_pos = len(data_arr[class_labels == 1])
    num_neg = len(data_arr[class_labels == 0])
    w = np.ones(data_arr.shape[0])
    w[class_labels == 1] /= 2 * num_pos  # 样本权重初始化
    w[class_labels == 0] /= 2 * num_neg

    best_classfiers = []
    for _ in range(num_it):
        w /= sum(w)  # weights normalized
        best_classfier = best_weak_classifier(data_arr, class_labels, w)
        beta = best_classfier['error'] / (1 - best_classfier['error'])  # 计算分类错误率
        error = best_classfier['error_flag']  # 是否分类错误
        w *= (beta ** (1 - error))  # update sample weights
        best_classfier['alpha'] = math.log(1 / (beta + 0.01))  # 计算弱分类器的权重并记录
        best_classfiers.append(best_classfier)  # 记录弱分类器

    # print(w)

    def strong_classfiers(new_data_arr):
        left = sum(
            [c['alpha'] * stump_classifier(new_data_arr, c['feature'], c['thresh_val'], c['thresh_ineq']) for c in
             best_classfiers])
        right = 1 / 2 * sum([c['alpha'] for c in best_classfiers])
        strong_predict = ((left - right) >= 0).astype(int)
        # print(strong_predict)
        return strong_predict

    return strong_classfiers


def boosting_sintree(data_arr, class_labels):
    '''
    基于单层决策树的AdaBoost
    '''
    # 随机选择n1个数据组成D1，剩下的组成D1_left
    n1 = len(data_arr) // 2
    # random_index = np.array([True if np.random.random() < n1/len(data_arr) else False for _ in range(len(data_arr))])
    D1_index = np.ones_like(class_labels, dtype=bool)  # 有放回采样，所有数据对应为True为选择
    D1_index[np.random.choice(range(len(data_arr)), n1, replace=False)] = False
    # plt.scatter(data_arr[D1_index][:, 0], data_arr[D1_index][:, 1], c=class_labels[D1_index], s=200)
    # plt.show()

    # 训练第一个弱分类器
    classfier1 = best_weak_classifier(data_arr[D1_index], class_labels[D1_index],
                                      w=np.ones_like(class_labels[D1_index]))

    # 构造D2
    D1_left_pred_arr = stump_classifier(data_arr[~D1_index], classfier1['feature'], classfier1['thresh_val'],
                                        classfier1['thresh_ineq'])
    # print("D1_left class label: \n{} \n D1_left predict label: \n{}".format(class_labels[~D1_index],
    # D1_left_pred_arr))
    D1_left = data_arr[~D1_index]
    class_labels_d1_left = class_labels[~D1_index]
    predict_flag = D1_left_pred_arr == class_labels[~D1_index]  # 是否分类正确bool数组
    correct_index = np.array(range(len(D1_left)))[predict_flag]  # D1_left中分类正确的下标
    incorrect_index = np.array(range(len(D1_left)))[~predict_flag]  # D1_left中分类不正确的下标
    n2 = n1 // 2  # ?D2数据数目
    D2_index = np.hstack([np.random.choice(correct_index, n2 / 2),
                          np.random.choice(incorrect_index, n2 / 2)])  # 从正确和错误分类中分别随机取出需要数据的一半，D2_index对应的是D1_left的下标
    # print(D2_index)

    # 训练第二个弱分类器
    classfier2 = best_weak_classifier(D1_left[D2_index], class_labels_d1_left[D2_index],
                                      w=np.ones_like(class_labels_d1_left[D2_index]))

    # 构造D3=D1_left[~D2_index]训练第三个弱分类器
    classfier3 = best_weak_classifier(D1_left[~D2_index], class_labels_d1_left[~D2_index],
                                      w=np.ones_like(class_labels_d1_left[~D2_index]))

    def strong_classfiers(new_data_arr):
        predict1 = stump_classifier(new_data_arr, classfier1['feature'], classfier1['thresh_val'],
                                    classfier1['thresh_ineq'])
        predict2 = stump_classifier(new_data_arr, classfier2['feature'], classfier2['thresh_val'],
                                    classfier2['thresh_ineq'])
        predict3 = stump_classifier(new_data_arr, classfier3['feature'], classfier3['thresh_val'],
                                    classfier3['thresh_ineq'])
        # print("p1:\n{}\np2:\n{}\np3:\n{}\n".format(predict1, predict2, predict3))
        predict1[predict1 != predict2] = predict3[predict1 != predict2]
        return predict1

    return strong_classfiers


if __name__ == '__main__':
    data_arr, class_labels = loadSimpData()
    print("origin data: \n {}\norigin data labels:\n{}".format(data_arr, class_labels))
    plt.scatter(data_arr[:, 0], data_arr[:, 1], c=class_labels)
    for i, class_label in enumerate(class_labels):
        plt.annotate(class_label, (data_arr[:, 0][i], data_arr[:, 1][i]))
    plt.show()

    strong_predict = boosting_sintree(data_arr, class_labels)(data_arr)
    print("strong_predict for boosting: \n{}".format(strong_predict))

    strong_predict = adaboost_sintree(data_arr, class_labels, num_it=9)(data_arr)
    print("strong_predict for adaboost: \n{}".format(strong_predict))
