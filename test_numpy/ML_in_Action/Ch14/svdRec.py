#coding:gbk
'''
Created on Mar 8, 2011

@author: Peter
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]  #7*5
    
def loadExData2():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 5, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
    
def ecludSim(inA,inB): #欧氏相似度
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):#皮尔逊线性相关相似度
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):#余弦相似度
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item):#在给定相似度计算方法的条件下，用户对物品的估计评分值(即该用户目前未对该物品打过分)
    n = shape(dataMat)[1]  #物品个数
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue #若找到当前用户对某个物品的打分不为零，item为需要评分的物体,使用的是基于物品的相似度
        overLap = nonzero(logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0 #如果两个人没有共同的物品打分，则相似度为0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])#否则计算它们的相似度
        #print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity #
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal  #求它们基于相似度的加权平均评分
    
def svdEst(dataMat, user, simMeas, item):#基于SVD的评分估计,item为需要评分的物体,使用的是基于物品的相似度
    n = shape(dataMat)[1]#物品个数
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat) #m*n
    Sig4 = mat(eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix，妙，将行向量转为对角矩阵
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):#有点像PCA白化的意思
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]#find unrated items ，寻找未评级的物品
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)#用户对该种未评分的物品进行分值估计
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]  #寻找前N个未评级的物品

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print 1,
            else: print 0,
        print ''

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print "****original matrix******"
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k] #对角阵
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:] #取U前numSV列，VT前numSV行
    print "****reconstructed matrix using %d singular values******" % numSV
    printMat(reconMat, thresh)
    
    print "***二者之差***"
    printMat(reconMat-myMat)
    print "误差之和："+str(sum(abs(reconMat-myMat)>thresh))
    
if __name__=='__main__':
    print "测试SVD:\n"
    Data=loadExData()
    U,Sigma,VT=linalg.svd(Data)
    print "\nSigma:\n"
    print Sigma
    
    sig3=mat([[Sigma[0],0,0],#只取前三个sigma
              [0,Sigma[1],0],
              [0,0,Sigma[2]]]) 
    reconData=U[:,:3]*sig3*VT[:3,:]
    print reconData
    
    myMat=mat(loadExData()) #7*5
    print "测试欧氏相似度：\n"
    print ecludSim(myMat[:,0], myMat[:,4])
    
    print "测试余弦相似度：\n"
    print cosSim(myMat[:,0], myMat[:,4])
    
    print "测试皮尔逊线性相似度：\n"
    print pearsSim(myMat[:,0], myMat[:,4])
    
    
    print "测试推荐：\n"
    myMat=mat(loadExData()) #7*5
    myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
    myMat[3,3]=2
    print myMat
    
    print "cosSim:"+str(recommend(myMat, 2)) #对第二个用户进行推荐,即矩阵第三行
    print "ecludSim:"+str(recommend(myMat, 2,simMeas=ecludSim)) #对第二个用户进行推荐,即矩阵第三行
    print "pearSim:"+str(recommend(myMat, 2,simMeas=pearsSim)) #对第二个用户进行推荐,即矩阵第三行
    
    print "测试用svd推荐"
    myMat=mat(loadExData2());
    U,Sigma,VT=la.svd(myMat)
    print "Sigma:\n"+str(Sigma)
    print sum(Sigma**2),sum(Sigma**2)*0.9
    print recommend(myMat, 1, estMethod=svdEst)
    
    print "图像压缩\n"
    imgCompress(2)
    
   