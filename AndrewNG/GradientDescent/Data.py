'''
Created on Apr 12, 2014

@author: pipi
'''
import re
import numpy as np

def loadData(filename):
    'load items into matrices : feature[ [1,...],... ]     target[]'
    feature = list()
    target = list()
    f = open(filename,'rb')
    for line in f:
        sample = re.split('\s+',line.strip())
        #check_list the dimension of each line,or ValueError: setting answer array element with a sequence.
        #print(len(sample))#need to delete the last line of file
        feature.append([1] + sample[0:-1])#construct x0 = 1
        target.append(sample[-1])
#     return np.mat(feature,np.float),np.mat(target,np.float)
    return np.array(feature,np.float),np.array(target,np.float)

