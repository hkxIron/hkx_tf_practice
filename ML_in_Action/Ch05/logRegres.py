# -*- coding: gbk -*-
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])#将整个列表当作元素添加到原列表中
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix,100*3
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix,100*1
    m,n = shape(dataMatrix)
    alpha = 0.001 #步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1)) #初始权重,3*1
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult ,矩阵相乘, 100*3,3*1
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult,3*100,100*1
    return weights

def plotBestFit(weights,_title='figure'):#画出数据集以及决策面对
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    plt.title(_title)
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s',label='positive')
    ax.scatter(xcord2, ycord2, s=30, c='green',label='negative')
    x = arange(-3.0, 3.0, 0.1) #从-3到3的数据集，其中步长为0.1
    y = (-weights[0]-weights[1]*x)/weights[2]  # w0+w1*x+w2*y=0
    y=array(y)
    x.shape=x.size,-1
    y.shape=x.size,-1
    
    ax.plot(x, y,c='black',label="line") #不知直线为何不能显示
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.legend()
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):#随机梯度上升
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m): #将所有样本训练一遍就结束
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * array(dataMatrix[i])
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix) #m行代表m个样本
    weights = ones(n)   #initialize to all ones
    for j in range(numIter): #循环迭代次数150次
        dataIndex = range(m)
        for i in range(m): #每一次迭代随机选取一个样本进行训练，直到所有样本训练完毕
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not go to 0 because of the constant
            randIndex = int(random.uniform(0,len(dataIndex)))#产生随机数
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h  #标签-预测值
            weights = weights + alpha * error * array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):#回归分类函数
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest(iters):
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):#0~20
            lineArr.append(float(currLine[i])) #将整个列表当作元素添加到原列表中
        trainingSet.append(lineArr) #将该行数据加入其中
        trainingLabels.append(float(currLine[21])) #最后一列为类标签
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000) #迭代1000次
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "第%d次：the error rate of this test is: %f" %(iters+1,errorRate)
    return errorRate

def multiTest():#多次运行然后求取平均值
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest(k)
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
 
print u"logistic回归测试"
dataArr,labelMat=loadDataSet()
weightsAsc=gradAscent(dataArr,labelMat)
print 'weights:',weightsAsc
# plotBestFit(weightsAsc,'gradAscent')


print u"随机梯度上升"
stocWeights=stocGradAscent0(dataArr,labelMat)
print 'stocWeights:',stocWeights
# plotBestFit(stocWeights,'stocGradAscent0')

print u"随机梯度上升,有衰减系数"
stocWeights1=stocGradAscent1(dataArr,labelMat)
print 'stocWeights:',stocWeights1
# plotBestFit(stocWeights1,'stocGradAscent1')

print u"分类马标签"
multiTest()
       