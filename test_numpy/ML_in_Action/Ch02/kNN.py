# -*- coding: gbk -*-
'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

    
#inX为输入向量，即待分类的向量，
#datsset为 4*2列，共4个点，每个点为2维
#labels共4个标签
#k为

#本例只是用穷举法来进行找最相邻的三个点，当数据很多时，会使用KD树来进行高效的查找

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]    #行数为4
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #tile 将inx [0,0]复制为4行1列
    sqDiffMat = diffMat**2  #逐元素平方，而不是矩阵相乘
    sqDistances = sqDiffMat.sum(axis=1) #按照第1维，即沿行方向，从左向右进行相加，
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  #升序排序   
    classCount={}          
    for i in range(k):#range(3)为 0,1,2,k=3,为3近邻
        voteIlabel = labels[sortedDistIndicies[i]] #选取与该点最近的三个点的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #如果标签键值不存在，则以0为默认值
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #按第2个属性进行逆序排序
    return sortedClassCount[0][0] #返回最大的频数的标签，即键名

#生成简单的数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#将文本记录到转换Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file，得到文件行数
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return，创建返回矩阵numpy
    classLabelVector = []                       #prepare labels return，创建返回标签   
    fr = open(filename)
    index = 0
    for line in fr.readlines():                 #readlines:一次读取文件的所有行
        line = line.strip()                     #截取两边所有回车与空格
        listFromLine = line.split('\t')         #原来是以tab键分割，但是文件中好像是3个空格键
#         i=0
#         for x in listFromLine:
#             if x==' ':
#                 del listFromLine[i]          #去除其中的空
#             i+=1
        
        returnMat[index,:] = listFromLine[0:3]  #索引0，1，2，不包括3,连float都不加，直接将string list转换成 float,完全交给numpy函数库处理
        classLabelVector.append(int(listFromLine[-1]))  #最后一个为类标签,为数字
        index += 1
    return returnMat,classLabelVector


#显示数据
def showData(dataSet,datingLabels):
#     plt.clf()
    fig=plt.figure()  
#     plt.ion()  #Switching interactive mode,这样可以不阻塞程序运行
    ax=fig.add_subplot(111)
    ax.scatter(dataSet[:,1],dataSet[:,2],
               15.0*array(datingLabels),15.0*array(datingLabels))
    plt.xlabel("game time")
    plt.ylabel("ice cream")
#     plt.show(block=False) 
    plt.draw()  #此处只画，但不显示
    
 
   
#归一化特征值    
def autoNorm(dataSet):
    minVals = dataSet.min(0)  #沿着各列，从上到下求最小值，n行3列
    maxVals = dataSet.max(0)  #
    ranges = maxVals - minVals  #最大值与最小值之差
    normDataSet = zeros(shape(dataSet)) 
    m = dataSet.shape[0]  #numpy矩阵,获得矩阵的行数
    normDataSet = dataSet - tile(minVals, (m,1))  #对行进行复制，然后相减
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide,逐像素相除
    return normDataSet, ranges, minVals   #返回归一化后的数值，差值，最小值

#测试分类器效果   
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load dataset from file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]  #行数
    numTestVecs = int(m*hoRatio) #用来测试的样例数
    errorCount = 0.0
    print "正在进行分类..."
    for i in range(numTestVecs):   #第i个作为测试           从第numTestVecs开始到最后，作为数据库           3：3近邻
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0 #分类错误，进行累加
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount
#预测分类  
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(\
                                  "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream, ]) #也可写为array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - \
                                  minVals)/ranges, normMat, datingLabels, 3)  #归一输入数值
    print "You will probably like this person: %s" % resultList[classifierResult - 1]  
    
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32): #共32行
        lineStr = fr.readline() #每次只读取文件的一行
        for j in range(32): #共32列
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    print "正在运行分类..."
    hwLabels = []  
    trainingFileList = listdir('trainingDigits')           #load the training set,listdir可以从指定目录中列出文件名
    m = len(trainingFileList)  #获取文件列表的长度
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr) #加入标签
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
        
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "%d:the classifier came back with: %d, the real answer is: %d" % (i+1,classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
    print "分类完毕..."


#print "begin debug"
# group,labels = createDataSet()
# classify0([0,0],group,labels,3 )
# datingDataMat,datingLabels =file2matrix('datingTestSet2.txt')
# normat,ranges,minvals=autoNorm(datingDataMat) #归一化特征
# showData(normat,datingLabels)

# #测试
# datingClassTest()
# plt.show()  #最后调用show来显示即可
# 
# #预测
# classifyPerson()

#手写识别
testVector=img2vector('testDigits/0_13.txt')
# print testVector[0,0:31]
# print testVector[0,32:63]

handwritingClassTest()
       