import numpy as np
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    
    return retArray
    

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with line, j, lessThan
                errArr = mat(ones((m,1)))
                
                
                errArr[predictedVals == labelMat] = 0
                
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (line, threshVal, inequal, weightedError)
                if weightedError < minError:
                    print(weightedError)
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    for i in bestStump.items():
        print(i)
    return bestStump,minError,bestClasEst
def adaBoostTrainDS( dataArr, classLabels, numIt = 40 ):
    '''
    基于单层决策树的AdaBoost训练过程
    '''
    weakClfArr = []
    m = np.shape( dataArr )[ 0 ]
    D = np.mat( np.ones( ( m, 1 ) ) / m )
    aggClassEst = np.mat( np.zeros( ( m, 1 ) ) )
    for i in range( numIt ):
        # 每一次循环 只有样本的权值分布 D 发生变化
        bestStump, error, classEst = buildStump( dataArr, classLabels, D )
        print(" D: ", D.T)

        # 计算弱分类器的权重
        alpha = float( 0.5 * np.log( ( 1 - error ) / max( error, 1e-16 ) ) )
        bestStump[ 'alpha' ] = alpha
        weakClfArr.append( bestStump )
        print("classEst: ", classEst.T)

        # 更新训练数据集的权值分布
        expon = np.multiply( -1 * alpha * np.mat( classLabels ).T, classEst )
        D = np.multiply( D, np.exp( expon ) )
        D = D / D.sum()

        # 记录对每个样本点的类别估计的累积值
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)

        # 计算分类错误率
        aggErrors = np.multiply( np.sign(aggClassEst) !=
            np.mat( classLabels ).T, np.ones( ( m, 1 ) ) )
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate, "\n")

        # 如果完全正确，终止迭代
        if errorRate == 0.0:
            break
    return weakClfArr

if __name__ == '__main__':
    print(__doc__)
    datMat, classLabels = loadSimpData()
#    plt.scatter(datMat[:, 0], datMat[:, 1], c=classLabels, markers=classLabels, s=200, cmap=plt.cm.Paired)
    print(adaBoostTrainDS( datMat, classLabels, 9 ))
