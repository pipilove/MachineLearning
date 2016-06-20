#coding=gbk
'''
Created on Apr 12, 2014

@author: pipi
'''
import numpy as np


def bgd(feature,target,alpha = 0.001,iterateTimes = 200):
    '... batch gradient descent ...'
    theta = np.zeros(feature.shape[1])#num of theta = num of feature atrribute
    #theta = np.ones(feature.shape[1])
#     print(theta.shape)
#     print(feature[0].shape)
    for it in range(iterateTimes):  #max iteratetimes is 200
        for i in range(feature.shape[0]):   #for each sample
            error = target[i] - sum(feature[i]*theta)#对应元素相乘，都是行array
            theta += alpha*error*feature[i]
         
        predict = [sum(theta*sample) for sample in feature] 
        mse = sum((predict - target)**2)/feature.shape[0]  
#       mse = sum((feature*theta - target)**2)/feature.shape[0]
#         print it,'mse:',mse
#         if(mse < 0.0001):
#             break
    print 'bgd_mse : ',mse
    return theta


def sgd(feature,target,alpha = 0.001,iterateTimes = 101000):#101000
    '... stochastic gradient descent ...'
    theta = np.zeros(feature.shape[1])#num of theta = num of feature atrribute
#     print(theta.shape)
#     print(feature[0].shape)
    for it in range(iterateTimes):  #max iteratetimes is 200
        i = it%feature.shape[0]
        error = target[i] - sum(feature[i]*theta)#对应元素相乘，都是行array
        theta += alpha*error*feature[i]
         
        predict = [sum(theta*sample) for sample in feature] 
        mse = sum((predict - target)**2)/feature.shape[0]  
#         print it,'mse:',mse
        if(mse < 21.8498395893):
            break
    print 'sgd_mse : ',mse
        
    return theta


def normalizer(feature):
    'normalization of feature'
    mean_j = np.mean(feature,axis = 0)#axis isn't exist
    #mat.mean()return mat[[]];array.mean()return array[]
#     print 'mean_j\n',mean_j
    std_j  = np.std(feature,axis = 0)
#     print 'std_j\n',std_j
    
#     print(feature)
    for j in range(1,feature.shape[1]):
        feature[:,j] = (feature[:,j] - mean_j[j])/std_j[j]#array - float!!!
#     print(feature)
        
    return feature

            