# coding: gbk
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''

from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats,delimited:限制，定…的界
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1])) #最后一列为特征
    return dataMat,labelMat

#分类函数
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1)) #retArray:5*1,初始化为+1
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D,*printOut):#D为各个样本最初的权重
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions，在所有维上循环
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();#取该维上的最大值与最小值
        stepSize = (rangeMax-rangeMin)/numSteps #计算该维上的步长
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension,此维上各阈值处
            for inequal in ['lt', 'gt']: #go over less than and greater than,在此维上此值处比较大于和小于
                threshVal = (rangeMin + float(j) * stepSize) #得到分割的阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0 #将预测值与标签相同的地方，即预测正确的设为0,错误的仍然为1
                weightedError = D.T*errArr  #calc total error multiplied by D,计算总的错误权重
                if printOut:
                    print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError: #记录最佳值
                    minError = weightedError
                    bestClasEst = predictedVals.copy() #最好的预测结果
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40,*printOut):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal,初始的权重
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump,每次循环时都将所有数据进行重分类，只是此时的数据权重不同
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0，错误越小，该分类器的权重alpha越大
        bestStump['alpha'] = alpha  #记录权重
        weakClassArr.append(bestStump)                  #store Stump Params in Array,用list记录这些分类器
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy, Multiply为逐元素相乘
        D = multiply(D,exp(expon))                              #Calc New D for next iteration, Adaboost本身更新权重的公式
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst #aggClassEst:为记录每个数据点的类别估计值
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1))) #只保留不一致的类标
        errorRate = aggErrors.sum()/m
        if printOut:
            print "total error: ",errorRate
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr,*printOut):#adaboost的分类函数
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst  #权重乘以分类标签
        if printOut:
            print aggClassEst
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):#画出ROC曲线,根据预测强度
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0) #正例个数
    yStep = 1/float(numPosClas);  #y轴以正例个数倒数为步长
    xStep = 1/float(len(classLabels)-numPosClas) #以负例个数倒数为步长
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0: #若为正例，对真阳率进行修改
            delX = 0; delY = yStep;
        else: #若为反例，对假阳率进行修改
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b') #plot(x,y),前面是X轴，后面是y轴
        cur = (cur[0]-delX,cur[1]-delY) #更新当前点的值
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep


if __name__ == "__main__":
    print u"单层决策树..."
    datMat,classLabels=loadSimpData()
    D=mat(ones((5,1))/5)
    bestStump,error,classEst =buildStump(datMat,classLabels,D)
    print u"最优决策树:"
    print bestStump,'\n',error,'\n',classEst 
    print u"AdaBoost训练..."
    classifierArr,aggClassEst=adaBoostTrainDS(datMat,classLabels,9,True)
    print u"AdaBoost分类..."
    print adaClassify([0,0],classifierArr)
    print adaClassify([5,5],classifierArr)
    
    print u"在马疝病数据集上测试Adaboost算法..."
    datArr,labelArr=loadDataSet('horseColicTraining2.txt')
    trainErrArr=mat(ones((len(labelArr),1)))
    trainNum=10
    classifierArray,aggClassEst=adaBoostTrainDS(datArr,labelArr,trainNum)
    errcount=trainErrArr[sign(aggClassEst)!=mat(labelArr).T].sum()
    print u"训练数据...："
    print u"训练次数",trainNum,"训练样本数目",len(labelArr)," 错误数:",errcount,"错误率：",errcount/len(labelArr)
    testArr,testLabelArr=loadDataSet('horseColicTest2.txt')
    prediction=adaClassify(testArr,classifierArray)
    errArr=mat(ones((len(testLabelArr),1)))
    errcount=errArr[prediction!=mat(testLabelArr).T].sum()
    print u"测试数据..."
    print u"测试样本数目",len(testLabelArr)," 错误数:",errcount,"错误率：",errcount/len(testLabelArr)
    
    print u"画出ROC曲线..."
    plotROC(aggClassEst.T,labelArr)
    
