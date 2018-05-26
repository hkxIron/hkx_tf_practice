#coding:gbk
'''
Created on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
'''
from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float(),使用映射
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1

def regLeaf(dataSet):#returns the value used for each leaf
    return mean(dataSet[:,-1])

def regErr(dataSet):
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]
    #if all the target variables are the same value: quit and return value
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1: #exit cond 1
        return None, leafType(dataSet) #如果所有样例具有相同值(类标)，则退出
    m,n = shape(dataSet)
    #the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n-1): #遍历所有特征
        for splitVal in set(dataSet[:,featIndex].A1):#遍历该特征在数据集上的所有取值，加了A1,或用.T.tolist()，set的取值才有效
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)#按该值进行划分数据集
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue #划分的样例数太少，直接跳过
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: #如果总方差较小，则认为较好，并记录下来
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS: #如果数据集中方差减小得并不明显，则退出
        return None, leafType(dataSet) #exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)#按最好特征划分，样例数仍较少
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #exit cond 3
        return None, leafType(dataSet)
    return bestIndex,bestValue#returns the best feature to split on
                              #and the value used for that split

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree,testData):#求取左右子树均值
    if isTree(tree['right']): tree['right'] = getMean(tree['right'],testData)
    if isTree(tree['left']): tree['left'] = getMean(tree['left'],testData)
    lN=0;rN=0
       
    if testData.size!=0: 
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        lN=shape(lSet)[0];rN=shape(rSet)[0]
   
    return (tree['left']*lN+tree['right']*rN)/(rN+lN+1e-10)
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree,testData) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet) #左子树减枝
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet) #右子树减枝
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']): #如果左右孩子均为叶子，看是否能合并
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2)) #tree['left']是一个键值，为左子树的平均值
        #treeMean = (tree['left']+tree['right'])/2.0
        lN=shape(lSet)[0];rN=shape(rSet)[0]
        treeMean=(lN*tree['left']+rN*tree['right'])/(lN+rN+1e-10)
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean #其实是一个均值
        else: return tree
    else: 
        return tree
#回归树计算    
def regTreeEval(model, inDat):
    return float(model)
#模型树计算
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)
#树的预测
def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat

if __name__=='__main__':

    print u'测试cart回归树..'
    myDat=loadDataSet('ex00.txt')
    myMat=mat(myDat)
#     tree0=createTree(myMat)
#     print tree0
    
    print u'测试cart回归树1..'
    myDat1=loadDataSet('ex0.txt')
    myMat1=mat(myDat1)
    tree1=createTree(myMat1)
    print tree1
    
    print u'测试剪枝..'
    myDat2=loadDataSet('ex2.txt')
    myMat2=mat(myDat2)
    tree2=createTree(myMat2,ops=(0,1)) 
    print u'将会产生庞大的树，产生过拟合。\n',tree2
    
    print u'进行后剪枝...'
    myDatTest=loadDataSet('ex2test.txt')
    myMat2Test=mat(myDatTest)
    treePruned=prune(tree2,myMat2Test)
    print u'后剪枝树：\n',treePruned
    
    print u'测试模型树...'
    myData=loadDataSet('exp2.txt')
    myMat2=mat(myData)
    modelTree= createTree(myMat2,modelLeaf,modelErr,(1,10))
    print u"模型树:",modelTree
    
    print u'比较模型树与回归树...'
    trainMat=mat(loadDataSet('bikeSpeedVsIq_train.txt'))
    testMat=mat(loadDataSet('bikeSpeedVsIq_test.txt'))
    myTree=createTree(trainMat,ops=(1,20)) #创建回归树
    yHat=createForeCast(myTree,testMat[:,0]) #进行预测
    print u'回归树相关系数:',corrcoef(yHat,testMat[:,1],rowvar=0)[0,1] #计算相关系数
    
    myTree=createTree(trainMat,modelLeaf,modelErr,ops=(1,20)) #创建模型树
    yHat=createForeCast(myTree,testMat[:,0],modelTreeEval) #进行预测
    print u'模型树相关系数:',corrcoef(yHat,testMat[:,1],rowvar=0)[0,1] #计算相关系数
    
    print u'线性回归：'
    ws,X,Y=linearSolve(trainMat)
    print ws
    for i in range(shape(testMat)[0]):
        yHat[i]=testMat[i,0]*ws[1,0]+ws[0,0]
    print u"线性回归相关系数:",corrcoef(yHat,testMat[:,1],rowvar=0)[0,1] 
    