#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '决策树多分类：使用scikit-learn实现'
__author__ = '卡卡'
__mtime__ = '12/11/2015-011'
__email__ = '1530818701@qq.com'
references:
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree
.DecisionTreeClassifier
http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
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
from random import random
from sklearn import tree
from sklearn.externals.six import StringIO
import numpy as np
import pydot


class GlobalOptions():
    # 文件路径
    original_filename = r'E:\mine\study\数据挖掘\project\hw2data.txt'  # 原始数据
    preprocessed_all_filename = r'pre_data_all.txt'

    all_data_filename = r'E:\mine\c_workspace\DecisionTree\DecisionTr\DecisionTr\data-all.txt'  # 卡卡的数据
    trainning_data_filename = r'E:\mine\python_workspace\MachineLearning\DataMining\data-train.txt'
    test_data_filename = r'E:\mine\python_workspace\MachineLearning\DataMining\data-LanguageAnalysis.txt'

    # 绘图时使用参数
    # feature_names = ["肝气郁结证型系数", "热毒蕴结证型系数", "冲任失调证型系数", "气血两虚证型系数", "脾胃虚弱证型系数", "肝肾阴虚证型系数"]#dot绘图不能使用中文！
    feature_names = ["SH1", "SH2", "SH3", "SH4", "SH5", "SH6"]
    target_names = ['H1', 'H2', 'H3', 'H4']
    labels = ['1', '2', '3', '4']


def Preprocess(ori_filename, pre_filename):
    '''
    数据预处理：选择6个属性作为features以及去除存在空值的features
    '''
    with open(ori_filename, encoding='utf-8') as ori_file, open(pre_filename, 'w', encoding='utf-8') as pre_file:
        for lineid, line in enumerate(ori_file):
            X = line.split('\t')[0:6]
            Y = line.split('\t')[7]
            # print(X)
            # print(Y)
            if '' in X or Y == '':
                continue
            pre_file.write('\t'.join(X))
            pre_file.write('\t' + Y + '\n')


def SplitData(all_filename):
    '''
    将数据划分为训练数据和测试数据
    '''
    with open(all_filename, encoding='utf-8') as all_data_file:
        with open(GlobalOptions.trainning_data_filename, 'w', encoding='utf-8') as train_file, open(
                GlobalOptions.test_data_filename, 'w') as  test_file:
            next(all_data_file)
            for line in all_data_file:
                if random() <= 2 / 3:
                    train_file.write(line)
                else:
                    test_file.write(line)


def ReadData(filename):
    '''
    从文件中读取X,Y
    '''
    Xs = []
    Ys = []
    Y_dict = {'H1': '1', 'H2': '2', 'H3': '3', 'H4': '4'}
    with open(filename, encoding='utf-8') as file:
        next(file)  # 忽略第一行
        for line in file:
            X = line.strip().split()
            Y = X[-1]
            X = X[0:-1]
            Xs.append(X)
            Ys.append(Y_dict[Y])
    return Xs, Ys


def PlotDecisionTree(clf):
    '''
    绘制决策树，命令行中也可以这样绘制：dot -Tpdf decision_tree.dot -o decision_tree.pdf
    '''
    # with open("decision_tree.dot", 'w') as f:
    #     tree.export_graphviz(clf, out_file=f, feature_names=GlobalOptions.feature_names)

    dot_data = StringIO()
    # 彩色完整图
    tree.export_graphviz(clf, out_file=dot_data, feature_names=GlobalOptions.feature_names,
                         class_names=GlobalOptions.target_names, filled=True, rounded=True, special_characters=True,
                         label='robot')
    # 黑白有深度图
    # tree.export_graphviz(clf, out_file=dot_data, feature_names=GlobalOptions.feature_names,
    #                      class_names=GlobalOptions.target_names, rounded=True, special_characters=True, label='robot',
    #                      max_depth=5)

    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("decision_tree.png")


def Predict(clf):
    '''
    预测test文件中的class label,并计算准确率。还可以使用from sklearn.cross_validation import cross_val_score
    '''
    X, Y = ReadData(GlobalOptions.test_data_filename)

    print('accuracy = ', clf.score(X, Y))

    Y_Predict = clf.predict(X)
    # accuracy = sum(sum([Y_Predict == Y])) / len(Y)  # <=>clf.score(X, Y)
    # print('accuracy = ', accuracy)

    # 计算各类分类precision
    for true_label in GlobalOptions.labels:
        Y_Predict_ = Y_Predict[np.array(Y) == true_label]
        for predict_label in GlobalOptions.labels:
            print(sum(np.array(Y_Predict_) == predict_label), end=' ')
        print()


if __name__ == '__main__':
    # 数据预处理：选择6个属性作为features以及去除存在空值的features
    # Preprocess(GlobalOptions.original_filename, GlobalOptions.preprocessed_all_filename)

    # 将数据分为训练数据和测试数据
    # SplitData(GlobalOptions.all_data_filename)  # 离散数据
    SplitData(GlobalOptions.preprocessed_all_filename)  # 连续数据

    # 训练数据
    X, Y = ReadData(GlobalOptions.trainning_data_filename)
    # print(X[0:10], Y[0:10])

    # 建立多类分类决策树
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
    clf = clf.fit(X, Y)

    # 绘制决策树
    PlotDecisionTree(clf)

    # 预测决策树分类，并计算准确率
    Predict(clf)
