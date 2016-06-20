#coding=gbk
"""���ڣУã��㷨������ʶ��
@author:      pipi
@time:        2014.11.1
"""
from PIL import Image, ImageDraw
import numpy
import cv
import os
import sys

IMAGE_SIZE = (40,40)
def createDatabase(path, number):
    """��ָ����·��path�д���number������˳������������ͼƬ,��Ϊһ������"""
    imageMatrix = []
    for i in range(1,number+1):
        image = Image.open(path+'/'+str(i)+'.jpg')
        image = image.resize(IMAGE_SIZE) #��СͼƬ
        grayImage = image.convert('L')
        imageArray = list(grayImage.getdata())#ת��Ϊһ��һά���飬����������
        imageMatrix.append(imageArray)

    imageMatrix = numpy.array(imageMatrix)  #ת��Ϊ��ά������Ϊͼ������ֵ�������У�û��Ϊһ��ͼ
    #print imageMatix

    return imageMatrix

def eigenfaceCore(Matrix):
    """ͨ���õ���ͼ�����ѵ��������"""
    trainNumber, perTotal = numpy.shape(Matrix) #����ͼ��ĸ�������ÿ��ͼ��Ĵ�С

    """�����м���ƽ������"""
    meanArray = Matrix.mean(0) #0�����м���ƽ����1�����м���ƽ��

    """����ÿ��������ƽ�������Ĳ�"""
    diffMatrix = Matrix - meanArray

    """����Э�������C�����L"""
    #diffMatrixTranspose = numpy.transpose(diffMatrix) #����ת��
    diffMatrix = numpy.mat(diffMatrix)#�����������͵�����
    L = diffMatrix * diffMatrix.T #ʹ�˵õľ����С
    eigenvalues, eigenvectors = numpy.linalg.eig(L) #��������v[:,i]��Ӧ����ֵw[i]

    """����õ�������ֵ��������������˳��
        ��һ����������ֵ����1����ȡ��������"""
    eigenvectors = list(eigenvectors.T) #��Ϊ�������������ÿ����һ������������
                                        #������Ҫת�ú󣬱�Ϊһ��list,Ȼ��ͨ��pop������
                                        #ɾ�����е�һ�У�����任ת��ȥ
    for i in range(0,trainNumber):
        if eigenvalues[i] < 1:
            eigenvectors.pop(i)

    eigenvectors = numpy.array(eigenvectors) #�����޷�ֱ�Ӵ���һά�ľ���������Ҫһ���������
    eigenvectors = numpy.mat(eigenvectors).T


    """������������,Ҳ���Ǽ����C
        ���ֱ任�����˼������"""
    #print numpy.shape(diffMatrix)
    #print numpy.shape(eigenvectors)
    eigenfaces = diffMatrix.T * eigenvectors
    return eigenfaces

def recognize(testIamge, Matrix, eigenface):
    """testIamge,Ϊ����ʶ��Ĳ���ͼƬ
    MatrixΪ����ͼƬ���ɵľ���
    eigenfaceΪ������
       ����ʶ������ļ����к�"""

    """�����м���ƽ������"""
    meanArray = Matrix.mean(0) #0�����м���ƽ����1�����м���ƽ��

    """����ÿ��������ƽ�������Ĳ�"""
    diffMatrix = Matrix - meanArray

    """ȷ���������˺��ͼƬ��Ŀ"""
    perTotal, trainNumber = numpy.shape(eigenface)

    """��ÿ������ͶӰ�������ռ�"""
    projectedImage = eigenface.T * diffMatrix.T

    #print numpy.shape(projectedImage)
    """Ԥ�������ͼƬ������ӳ�䵽�����ռ���"""
    testimage = Image.open(testIamge)
    testimage = testimage.resize(IMAGE_SIZE)
    grayTestImage = testimage.convert('L')
    testImageArray = list(grayTestImage.getdata())#ת��Ϊһ��һά���飬����������
    testImageArray = numpy.array(testImageArray)

    differenceTestImage = testImageArray - meanArray
    #ת��Ϊ������ڽ������ĳ˷�����
    differenceTestImage = numpy.array(differenceTestImage)
    differenceTestImage = numpy.mat(differenceTestImage)

    projectedTestImage = eigenface.T * differenceTestImage.T
    #print numpy.shape(projectedImage)
    #print numpy.shape(projectedTestImage)

    """����ŷʽ���������ƥ�������"""
    distance = []
    for i in range(0, trainNumber):
        q = projectedImage[:,i]
        temp = numpy.linalg.norm(projectedTestImage - q) #���㷶��
        distance.append(temp)

    minDistance = min(distance)
    index = distance.index(minDistance)

    return index+1 #����index�Ǵ�0��ʼ��


if __name__ == "__main__":
    TrainNumber = len(os.listdir('./faces_extract')) - 1
    Matrix = createDatabase('./faces_extract', TrainNumber)
    eigenface = eigenfaceCore(Matrix)
    testimage = './faces_test/1.png'
    print(recognize(testimage, Matrix, eigenface))
