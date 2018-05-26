# cart_adaboost.py
# coding:utf8
from itertools import *
import operator,time,math
import matplotlib.pyplot as plt
def calGini(dataSet):#计算一个数据集的gini系数
    numEntries = len(dataSet)
    labelCounts={}
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini=1
    for label in labelCounts.keys():
        prop=float(labelCounts[label])/numEntries
        gini -=prop*prop
    return gini

def splitDataSet(dataSet, axis, value,threshold):#根据特征、特征值和方向划分数据集
    retDataSet = []
    if threshold == 'lt':
        for featVec in dataSet:
            if featVec[axis] <= value:
                retDataSet.append(featVec)
    else:
        for featVec in dataSet:
            if featVec[axis] > value:
                retDataSet.append(featVec)

    return retDataSet   
# 由于是连续值，如果还是两两组合的话，肯定爆炸
# 并且连续值根本不需要组合，只需要给定值，就可以从这个分开两份
# 返回最好的特征以及特征值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1     
    bestGiniGain = 1.0; bestFeature = -1;bsetValue=""
    for i in range(numFeatures):        #遍历特征
        featList = [example[i] for example in dataSet]#得到特征列
        uniqueVals = list(set(featList))       #从特征列获取该特征的特征值的set集合
        uniqueVals.sort()
        for value in uniqueVals:# 遍历所有的特征值
            GiniGain = 0.0
            # 左增益
            left_subDataSet = splitDataSet(dataSet, i, value,'lt')
            left_prob = len(left_subDataSet)/float(len(dataSet))
            GiniGain += left_prob * calGini(left_subDataSet)
            # 右增益
            right_subDataSet = splitDataSet(dataSet, i, value,'gt')
            right_prob = len(right_subDataSet)/float(len(dataSet))
            GiniGain += right_prob * calGini(right_subDataSet)
            # print GiniGain
            if (GiniGain < bestGiniGain):      #比较是否是最好的结果
                bestGiniGain = GiniGain      #记录最好的结果和最好的特征
                bestFeature = i
                bsetValue=value
    return bestFeature,bsetValue                  

def majorityCnt(classList):#多数表决
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 这里用做生成弱分类器的cart
def createTree(dataSet,depth=3):#生成一棵指定深度的cart
    classList = [example[-1] for example in dataSet]
    if depth==0:#如果到达指定深度，直接多数表决
        return majorityCnt(classList)
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#所有的类别都一样，就不用再划分了
    if len(dataSet) == 1: #如果没有继续可以划分的特征，就多数表决决定分支的类别
        return majorityCnt(classList)
    bestFeat,bsetValue = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=str(bestFeat)+":"+str(bsetValue)#用最优特征+阀值作为节点，方便后期预测
    if bestFeat==-1:
        return majorityCnt(classList)
    myTree = {bestFeatLabel:{}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = list(set(featValues))
    # print bsetValue
    myTree[bestFeatLabel]['<='+str(round(float(bsetValue),3))] = createTree(splitDataSet(dataSet, bestFeat, bsetValue,'lt'),depth-1)
    myTree[bestFeatLabel]['>'+str(round(float(bsetValue),3))] = createTree(splitDataSet(dataSet, bestFeat, bsetValue,'gt'),depth-1)
    return myTree  


def translateTree(tree,labels):
    if type(tree) is not dict:
        return tree
    root=tree.keys()[0]
    feature,threshold=root.split(":")#取出根节点，得到最优特征和阀值
    feature=int(feature)
    myTree={labels[feature]:{}}
    for key in tree[root].keys():
        myTree[labels[feature]][key]=translateTree(tree[root][key], labels)
    return myTree
def predict(tree,sample):
    if type(tree) is not dict:
        return tree
    root=tree.keys()[0]
    feature,threshold=root.split(":")#取出根节点，得到最优特征和阀值
    feature=int(feature)
    threshold=float(threshold)
    if sample[feature]>threshold:#递归预测
        return predict(tree[root]['>'+str(round(float(threshold),3))], sample)
    else:
        return predict(tree[root]['<='+str(round(float(threshold),3))], sample)

#用cart对数据集做预测，
def cartClassify(dataMatrix,tree):
    errorList=ones((shape(dataMatrix)[0],1))# 返回预测对或者错，而不是返回预测的结果(对为0，错为1，方便计算预测错误的个数)
    predictResult=[]#记录预测的结果
    classList = [example[-1] for example in dataSet]
    for i in range(len(dataMatrix)):
        res=predict(tree,dataMatrix[i])
        errorList[i]=res!=classList[i]
        predictResult.append([int(res)] )
        # print predict(tree,dataMatrix[i]),classList[i]
    return errorList,predictResult

#记录弱分类器，主要调整样本的个数来达到调整样本权重的目的，训练弱分类器由createTree函数生成
def weekCartClass(dataSet,weiths,depth=3):
    min_weights = weiths.min()#记录最小权重
    newDataSet=[]
    for i in range(len(dataSet)):#最小权重样本数为1，权重大的样本对应重复math.ceil(float(array(weiths.T)[0][i]/min_weights))次
        newDataSet.extend([dataSet[i]]*int(math.ceil(float(array(weiths.T)[0][i]/min_weights))))
    bestWeekClass={}
    dataMatrix=mat(dataSet);
    m,n = shape(dataMatrix)
    bestClasEst = mat(zeros((m,1)))
    weekCartTree = createTree(newDataSet,depth)
    errorList,predictResult=cartClassify(dataSet, weekCartTree)
    weightedError = weiths.T*errorList#记录分错的权重之和
    bestWeekClass['cart']=weekCartTree
    return bestWeekClass,predictResult,weightedError

def CartAdaboostTrain(dataSet,num=1,depth=3):
    weekCartClassList=[]
    classList = mat([int(example[-1]) for example in dataSet])
    m=len(dataSet)
    weiths=mat(ones((m,1))/m) #初始化所有样本的权重为1/m
    finallyPredictResult = mat(zeros((m,1)))
    for i in range(num):
        bestWeekClass,bestPredictValue,error=weekCartClass(dataSet,weiths,depth)#得到当前最优的弱分类器
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#根据error计算alpha
        bestWeekClass['alpha'] = alpha 
        expon = multiply(-1*alpha*mat(classList).T,bestPredictValue)
        weiths = multiply(weiths,exp(expon)) 
        weiths = weiths/weiths.sum()
        finallyPredictResult += alpha*mat(bestPredictValue)
        nowPredictError = multiply(sign(finallyPredictResult) != mat(classList).T,ones((m,1)))
        errorRate = nowPredictError.sum()/m
        print "total error: ",errorRate
        bestWeekClass['error_rate']=errorRate
        weekCartClassList.append(bestWeekClass)
        if errorRate == 0.0: break
    return weekCartClassList, finallyPredictResult


import treePlotter
from treePlotter import createPlot
import numpy as np
from numpy import *
filename="sample"
dataSet=[];labels=[];
with open(filename) as f:
    for line in f:
        fields=line.strip("\n").split("\t")
        t=[float(item) for item in fields[0:-1]]
        t.append(int(fields[-1]))
        dataSet.append(t)
labels=['x','y']
weekCartClass, finallyPredictResult=CartAdaboostTrain(dataSet,10,4)
print finallyPredictResult.T