# -*- coding: gbk -*-
'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m): #随机的选择一个与i不同的j
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):#裁剪aj在上下界之内
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):#简单的SMO算法，可以实现SVM
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose() #标签数据
    b = 0; m,n = shape(dataMatrix) #m个样本，n维
    alphas = mat(zeros((m,1))) #alpha维数与样本的个数相等，均为m个
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b #multiply为逐元素相乘
            #float(multiply(alphas,labelMat).T为alpha_i*y_i,1*m
            #dataMatrix*dataMatrix[i,:].T算的是所有向量与Xi的内积，即K(xi,xj),m*1
            #二者矩阵相乘，则为g(xi)=W*Xi+b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions,检测预测与实际的差值
            #若满足KKT条件，则 yi*g(xi)>=1,若差值过大，则进行调整alpha
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):#违反条件，需要调整
                #不明白为何上面语句就能判断违反KKT条件
                #0<alpha[i]<C,C=alpha[i]+mu[i]=>mu[i]>0=>cansi[i]=0,x[i]应落在间隔边界上
                #若labelaMat[i]=1,Ei<0,即fXi<labelMat[i],即并没有落间隔边界上，而是在正间隔边界与超平面之间，需要调整,或Ei>0,fXi>labelMat[i],即Xi在正向边界之外，并非支持向量
                #若labelaMat[i]=-1,Ei>0,fXi>labelMat[i],,即并没有落间隔边界上，而是在负间隔边界与超平面之间，需要调整,或Ei<0,fXi<labelMat[i],即Xi在负向边界之外，并非支持向量
                j = selectJrand(i,m) #j目前为随机选取
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b #g(xj)=W*Xj+b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();#进行深度复制
                if (labelMat[i] != labelMat[j]): #若二者符号相反
                    L = max(0, alphas[j] - alphas[i]) #求下界
                    H = min(C, C + alphas[j] - alphas[i]) #求上界
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                #eta=K11+K22-2K12,其中Kij=K(xi,xj),公式参见李航SVM
                eta =  dataMatrix[i,:]*dataMatrix[i,:].T + dataMatrix[j,:]*dataMatrix[j,:].T-2.0 * dataMatrix[i,:]*dataMatrix[j,:].T
                if eta <= 0: print "eta<=0"; continue #应该不会为负，因为 K11+K12_2K12肯定非负
                alphas[j] += labelMat[j]*(Ei - Ej)/eta #alpha_new=alpha_old+y2*(Ei-Ej)/eta;
                alphas[j] = clipAlpha(alphas[j],H,L) #在有效范围内裁剪
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                                                                        #因为 delta_alpha1*y1+delta_alpha2*y2=0
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0 #若二者间有为0或者为C，则取二者中点
                alphaPairsChanged += 1 #遗漏了对Ei的更新
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0 # 只有在所有数据上遍历maxIter次，且不再发生任何alpha修改之后，程序才会退出while循环
        print "iteration number: %d" % iter
    return b,alphas
#核转换函数
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel #线性核，即普通内积
    elif kTup[0]=='rbf': #径向基函数
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab,kTup为sigma
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn #待分类数据
        self.labelMat = classLabels #类别标签
        self.C = C #C值
        self.tol = toler #误差的允许值
        self.m = shape(dataMatIn)[0] #样本的个数
        self.alphas = mat(zeros((self.m,1))) #
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag,误差缓存函数
        self.K = mat(zeros((self.m,self.m))) 
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b) # Yi=W*Xi+b,预测值
    Ek = fXk - float(oS.labelMat[k]) #预测值-实际值=残差
    return Ek
        
def selectJ(i, oS, Ei): #this is the second choice -heurstic, and calcs Ej,启发式搜索
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E,最大误差的点,其中第一位表示是否有效
    validEcacheList = nonzero(oS.eCache[:,0].A)[0] #其中oS.eCache[:,0].A，是将矩阵转化为数组，行列大小都未改变,返回非零E值所对的索引值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej #线性扫描最大的误差点
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m) #随便选择一个
        Ej = calcEk(oS, j)
    return j, Ej 

def updateEk(oS, k):#after any alpha has changed update the new value in the cache, 更新最大误差列表
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
   
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand,启发式搜索alpha
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel #计算eta
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache，更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1 #若有任意一对alpha发生变化，那么返回1
    else: return 0
#完整个的Platt SMO算法优化程序外循环     
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):#当迭代次数达到指定时或遍历整个集合都未对alpha进行修改时，就退出循环
        alphaPairsChanged = 0
        if entireSet:   #go over all，遍历所有
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas #遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] #0<alpha<C应为在边界上的支持向量才对啊
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):#计算w超平面
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w
#测试径向基函数
def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0] #支持向量为那些alpha>0的向量 
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m)    
#来自第2章的KNN    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m) 


'''#######********************************
Non-Kernel VErsions below
'''#######********************************

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

def drawSVMPoints(pts,spts,w,_title='figure'):#画出SVM点以及决策平面, 分别为非支持向量点，技持向量，w为分界平面
    import matplotlib.pyplot as plt
    pts=array(pts);spts=array(spts);
    fig = plt.figure()
    plt.title(_title)
    ax = fig.add_subplot(111)
    ax.scatter(pts[:,0], pts[:,1], s=30, c='blue', marker='o',label='non-support points')
    ax.scatter(spts[:,0], spts[:,1], s=40, c='red', marker='s',label='support points')
    xMin=min(hstack((pts[:,0],spts[:,0]))) #按列方向进行组合, 或concatenate((a, b), axis=0)
    xMax=max(hstack((pts[:,0],spts[:,0])))
    x = array((xMin, xMax)) #从-3到3的数据集, 画直线两端点
    y = (-w[0]-w[1]*x)/w[2]  # w0+w1*x+w2*y=0
    y=array(y)
    x.shape=x.size,-1
    y.shape=x.size,-1
    ax.plot(x, y,c='black',label="line") #不知直线为何不能显示
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.legend()
    plt.show()

#测试id,发现python传递的所有参数都是引用传递
def testRef(x):    
     print u"函数内部id:%d,%s"%(id(x),x.__class__)

x=100;print u"外部id=",id(x),
print x.__class__;
testRef(x)
x=range(10);print u"外部id=",id(x),
print x.__class__;
testRef(x)

#print u"测试简单的SMO"
dataArr,labelArr=loadDataSet('testSet.txt')
'''
b,alphas=smoSimple(dataArr,labelArr,0.6,0.001,40)
print u'b:',b
alphaSupport=alphas[alphas>0]
print u'支持向量的alpha:',alphaSupport
print u'支持向量个数：',shape(alphaSupport)
print u'输出支持向量:'
pts=[];spts=[];
for i in range(100):
    if alphas[i]>0:
        print dataArr[i],labelArr[i];
        spts.append(dataArr[i]) #这里是list，而不是numpy，所以不能用dataArr[i,:]
    else:
        pts.append(dataArr[i])
labelMat=mat(labelArr)
alphaY=multiply(alphas[alphas>0],labelMat[alphas.T>0])
dataMat=mat(dataArr)[tile(alphas>0,(1,2))].reshape(-1,2)
w= multiply(tile(alphaY,(2,1)),dataMat.T)
w=w.sum(axis=1) #按照第1维即列相加
w=vstack((b,w))
drawSVMPoints(pts,spts,w,'svm')
'''

print u"完整版的svm:"
b,alphas=smoP(dataArr,labelArr,0.6,0.001,40)
print u'b:',b
alphaSupport=alphas[alphas>0]
print u'支持向量的alpha:',alphaSupport
print u'支持向量个数：',shape(alphaSupport)
print u'输出支持向量:'
pts=[];spts=[];
for i in range(100):
    if alphas[i]>0:
        print dataArr[i],labelArr[i];
        spts.append(dataArr[i]) #这里是list，而不是numpy，所以不能用dataArr[i,:]
    else:
        pts.append(dataArr[i])
labelMat=mat(labelArr)
alphaY=multiply(alphas[alphas>0],labelMat[alphas.T>0])
dataMat=mat(dataArr)[tile(alphas>0,(1,2))].reshape(-1,2)
w= multiply(tile(alphaY,(2,1)),dataMat.T)
w=w.sum(axis=1) #按照第1维即列相加
w=vstack((b,w))
drawSVMPoints(pts,spts,w,'svm-smo')

print u'测试Rbf函数:'
testRbf();

print u"数字识别..."
testDigits(('rbf',10))
