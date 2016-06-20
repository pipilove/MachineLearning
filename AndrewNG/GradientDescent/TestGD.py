'''
Created on Apr 12, 2014

@author: pipi
'''
from Data import loadData
from GradientDescent.BGD import normalizer, bgd, sgd


def testGD():
    (x,y) = loadData(r'E:\machine_learning\datasets\housing_data\housing_data.txt')
    x = normalizer(x)
    theta_bgd = bgd(x,y)
    print 'theta_bgd : \n',theta_bgd
    theta_sgd = sgd(x,y)
    print 'theta_sgd : \n',theta_sgd
    
    
if __name__ == '__main__':
    print('loading ... ')
    testGD()
    print('ending ... ')
    
    