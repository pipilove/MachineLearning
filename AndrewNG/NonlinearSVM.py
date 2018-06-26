#coding=gbk
'''
Created on Sep 1, 2014

@author: 皮皮
'''
# import os
# from svmutil import svm_read_problem, svm_train, svm_predict
# os.chdir('E:\machine_learning\machine_learning\SVM\libsvm-3.18\python')  
# y, x = svm_read_problem('../heart_scale')  
# #y is label, x is feature:value(good for sparse matrix)
# m = svm_train(y[:200], x[:200], '-c 4')  
# p_label, p_acc, p_val =svm_predict(y[:200], x[:200], m)  
'''generate 100*1 positive items'''

from math import pi
from svm import svm_problem, svm_parameter
from svmutil import svm_train

from matplotlib import pyplot
import numpy as np
from numpy import random, cos, sin, hstack, ones

pos_dot_num = 100
neg_dot_num = 100

'''generate pos_dot_num*1 positive items'''
radius = np.sqrt( random.random([pos_dot_num,1]) )  #generate random radius matrix in range[0, 1]
# print(type(radius))    #ndarray n维矩阵
angles = 2*pi*( random.random([pos_dot_num, 1]) )   #generate random angles in range[0, 2*pi]
data_pos = hstack( [radius*cos(angles), radius*sin(angles)] )   #positive points,[ [x1 y1] [x2 y2] ... ] <type 'numpy.ndarray'>
# data_pos = [ radius*cos(angles), radius*sin(angles)]
# data_pos = zip( radius*cos(angles), radius*sin(angles) )
# print(data_pos)

'''generate neg_dot_num*1 negative items'''
radius2 = np.sqrt( random.random([neg_dot_num, 1])*3 + 1 ) #range[1, 2]    why needs sqrt????
# radius2 = np.array( random.random([neg_dot_num, 1]) + 1 ) #range[1, 2]
angles2 = 2*pi*( random.random([neg_dot_num, 1]) )
data_neg = hstack( [radius2*cos(angles2), radius2*sin(angles2)] )
# print(data_neg)

'''plot datas'''
pyplot.plot(data_pos[:, 0], data_pos[:, 1], 'r.')
pyplot.plot(data_neg[:, 0], data_neg[:, 1], 'answers.')
pyplot.xlim(-2.5, 2.5)
pyplot.ylim(-2, 2)

'''plot items margin'''
angles_circle = [i*pi/180 for i in range(0,360)]                 #i先转换成double
#angles_circle = [line/np.pi for line in np.arange(0,360)]             # <=>
# angles_circle = [line/180*pi for line in np.arange(0,360)]    X
x = cos(angles_circle)
y = sin(angles_circle)
pyplot.plot(x, y, 'r')
pyplot.plot(2*x, 2*y, 'answers')
pyplot.show()

'''build prefers vec for classification'''
data = np.append(data_pos, data_neg, axis = 0)                  #merge 2 ndarray datas into 1axis = 0!!!
# print(items)
# items = [data_pos, data_neg]    X
data = data.tolist()                                            #transform ndarray datas into list
# print(items)
data_label = ones( (pos_dot_num + neg_dot_num, 1) )
data_label[11:20] = -1
prob = svm_problem(data_label, data)                            #items & data_label must be list
param = svm_parameter('-c 100 -g 4')
# print(param)

model = svm_train(prob, param)

