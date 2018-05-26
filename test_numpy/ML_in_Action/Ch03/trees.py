# -*- coding: gbk -*-
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''

from math import log
import operator

#创建简单的鉴定鱼集
def createDataSet():
    #   需要浮出水面么？    有脚噗   属于鱼类
    dataSet = [[1, 1, 'yes'], #创建list
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    #        需要浮出水面么？                 有脚噗
    labels = ['no surfacing','flippers'] # 脚蹼  ,list
    #change to discrete values
    return dataSet, labels

#计算数据值的香农熵,其实只计算了标签的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #计算数据集的行数，即实例个数
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]   #最后一列为标签
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 #如果当前标签不在其中，次数为0
        labelCounts[currentLabel] += 1 #可以使用, classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #如果标签键值不存在，则以0为默认值
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  #计算概率
        if prob!=0: #如果为0的话，直接跳过
            shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt 

#按照给定特征划分数据集  ,返加的是在该维axis上取该值value的不包括该维的样本子集  
def splitDataSet(dataSet, axis, value):
    retDataSet = [] #创建新的list对象
    for featVec in dataSet: #featVec为dataSet中的每一个子list，从第0个开始遍历，list: [1, 1, 'yes']
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting,不包括axis
            reducedFeatVec.extend(featVec[axis+1:]) # 列表可包含任何数据类型的元素，单个列表中的元素无须全为同一类型。 
            #append（）方法向列表的尾部添加一个新的元素。只接受一个参数。
            #extend()方法只接受一个列表作为参数，并将该参数的每个元素都添加到原有的列表中。
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的特征对数据集进行划分，即信息增益最大的特征进行划分    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels,最后一列为分类标签,共2个特征
    baseEntropy = calcShannonEnt(dataSet)  #计算标签的熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features,遍历所有特征
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature,featList为对所有样本进行该维特征的提取,比如所有样本脚噗特征的取值
        uniqueVals = set(featList)       #get a set of unique values,集合可以去掉重复值
        newEntropy = 0.0
        for value in uniqueVals: #遍历特征值取值的组合
            subDataSet = splitDataSet(dataSet, i, value) #在该维上对数据集进行划分,list: [[1, 'no'], [1, 'no']]
            prob = len(subDataSet)/float(len(dataSet)) #样本子集/所有样本
            newEntropy += prob * calcShannonEnt(subDataSet) #计算该特征对数据集的条件熵    
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy,计算信息增益
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i  #计算信息增益最大的特征
    return bestFeature                      #returns an integer
#对该数据集中出现最多的标签进行统计，即标签类
def majorityCnt(classList):
    classCount={} #类标签数,创建的是字典，而不是列表
    for vote in classList: #所有的类标签
        if vote not in classCount.keys(): classCount[vote] = 0 #用类标签作为键
        classCount[vote] += 1 #标签值加1，计算该标签出现的次数
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
#创建树的代码,返回值类型为字典
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #获取分类类别列表
    if classList.count(classList[0]) == len(classList): #若所有类别标签都相同，则返回该标签，‘yes’或者‘no’，即只是一个叶结点
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList) #若列只有一维，即只有分类标签，则没有特征，则停止分割，并返回类别最多的类
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最好的特征对数据集进行划分，即信息增益最大的特征进行划分    
    bestFeatLabel = labels[bestFeat] #最好的特征字符串:'no surfacing','flippters'
    myTree = {bestFeatLabel:{}} #字典
    del(labels[bestFeat]) #删除该特征标签字符串，即只剩下‘flippters’
    featValues = [example[bestFeat] for example in dataSet] #取得所有样本该维特征的取值
    uniqueVals = set(featValues) #去重
    for value in uniqueVals: #遍历特征取值，在python中，当参数为列表类型时，是按照引用方式传递的，为了保证每次调用不改变原列变内容，必须使用新变量sublabels代替原始列表
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)  #递归调用
    return myTree                            
#使用决策树的分类函数    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0] #no surfacing,获取树的根结点的名字
    secondDict = inputTree[firstStr]#通过键来获取值,其实是颗子树
    featIndex = featLabels.index(firstStr) #通过inputTree来获取到底现在要比较哪一个特征
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat #如果是叶结点，直接返回
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
#调试用
dataset,labels=createDataSet()
print dataset
print labels
ent=calcShannonEnt(dataset)
print u"熵:",ent

print u"按照给定特征划分数据集"
# print splitDataSet(dataset,0,1)
# print splitDataSet(dataset,0,0)

feature=chooseBestFeatureToSplit(dataset)
print u"best feature:",feature

print u"创建树"
myDat,labels=createDataSet()
myTree=createTree(myDat,labels)
print myTree
#dict: {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

import treePlotter
# treePlotter.createPlot0()
myDat,labels=createDataSet()
myTree=treePlotter.retrieveTree(0)
print myTree
print classify(myTree,labels,[1,0])
print u"存储树..."
storeTree(myTree,'classifierStorage.txt')
print u"读取树..."
print grabTree('classifierStorage.txt')

print u"\n隐形眼镜:"
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines() ]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=createTree(lenses,lensesLabels)
print lensesTree

treePlotter.createPlot(lensesTree)




