#coding: utf-8
'''
Created on Jul 13, 2014

@author: 皮皮
'''

import matplotlib.pyplot as plt  
import numpy as np
import math

# def mse(list_x, list_y):
#     mse = 0.0
#     for x, y in zip(list_x, list_y):
#         mse += math.pow(x -y, 2)
#     return mse / len(list_x)       
               

if __name__ == '__main__':    
    try:
        #numpy的loadtxt方法可以直接读取文本数据到numpy二维数组, 不够则一维
        txtArray = np.loadtxt(r"E:\machine_learning\datasets\housing_data\housing_data_years_price.txt")
        years = txtArray[ : ,0]
        prices = txtArray[ : ,1]
        plt.plot(years, prices, 'o')
         
        alpha = 0.001 #learning rate
        threshold = 0.00001
        theta0 = 2.0
        theta1 = 2.0
        new_error = 0.0
           
        while True:
            #迭代更新theta
            sum0 = 0.0
            sum1 = 0.0
            for year, price in zip(years, prices):
                tmp_sum = theta0 + theta1*year - price
                sum0 += tmp_sum
                sum1 += tmp_sum * year
            theta0 -= (sum0 / len(years)) * alpha #是同步更新的
            theta1 -= (sum1 / len(years)) * alpha
            
            #计算J函数误差
            old_error = new_error
            new_error = 0
            for year, price in zip(years, prices):
                new_error += math.pow(theta0 + theta1 * year - price, 2)
            new_error /= len(years) * 2     # /2m
            print(new_error)
             
            if( abs(new_error - old_error) < threshold):
                break
            
#         print('the hypothesis function is: y = %f + %fx' %(theta0, theta1))
        print 'The hypothesis function is:\n y =', theta0, ' + ', theta1, 'x'
        x = range(len(years))
        y = [ix*theta1 + theta0 for ix in x]
        plt.plot(x, y)
        plt.show()            
        
    except IOError:
        print('file open error!!!')
        