#coding:gbk
'''
Created on Jun 1, 2011

@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()] #strip()为去掉空格
    datArr = [map(float,line) for line in stringArr] #line->float
    return mat(datArr) #将list转换为mat

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0) #dataMat:N*D，即每一列为一个特征，每一行为一个数据
    meanRemoved = dataMat - meanVals #remove mean,在numpy中是自动记广播扩展维度
    covMat = cov(meanRemoved, rowvar=0) #covMat是ndarray
    eigVals,eigVects = linalg.eig(mat(covMat)) # inv(P)AP=Eig
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions,-1代表逆序查找
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest，取前k个特征向量,D*K,P_reduced=P(1:k)
    lowDDataMat = meanRemoved * redEigVects #transform data into new dimensions, X:n*D,新低维数据： y=X*P_reduced,维度为：n*K
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # x_hat=y*P.T+meanVals，将低维数据转置后加上平均值
    return lowDDataMat, reconMat #低维空间中的信号，重建后的信号

def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ') #1567行的点数据，每个点两个维度
    numFeat = shape(datMat)[1] #共有1567行数据，590维特征
    for i in range(numFeat): #对于每个维度里有nan的，先去掉，然后再求均值,datMat[:,i].A是指的ndarray，A1指的是一维数组
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number),nonzero返回的是索引数组tuple,tuple里有行与列共两个下标
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean ，将nan设为均值，将该列的所有nan的值同时设为meanVal
    return datMat

if __name__=='__main__':
    print u'测试 pca:\n'
    dataMat=loadDataSet('testSet.txt')
    lowDMat,reconMat=pca(dataMat,1) #可以设置成2，来测试是否正确
    print shape(lowDMat)
    import matplotlib
    import matplotlib.pyplot as plt 
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90) #
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red') #重建后的信号，如果使用了降维，那么重建后的信号是有误差的
    plt.show()
    
    print u'测试 replaceNanWithMean \n'
    dataMat=replaceNanWithMean()
    meanVals=mean(dataMat,axis=0);
    meanRemoved=dataMat-meanVals;
    covMat=cov(meanRemoved,rowvar=0); #590*590
    eigVals,eigVects=linalg.eig(mat(covMat)) #eigVals为特征值，eigVects是特征向量
    print "end"
    
    
    