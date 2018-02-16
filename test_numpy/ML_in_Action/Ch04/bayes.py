# -*- coding: gbk -*-
'''
Created on Oct 19, 2010

@author: Peter
'''
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

          
def createVocabList(dataSet):#创建一个包含在所有文档中出现的不重复词的列表       
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)
#输入的语料转化为词汇向量,词集模型
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) #长度为32的计数列表，因为共有32个单词
    for word in inputSet:
        if word in vocabList: #如果该单词在词汇表中，记录为1
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word  #该单词未被词汇表收录，则提示
    return returnVec #返回向量化后的文档语料

# 计算每个单词的类条件概率，trainMatrix:为文档矩阵，trainCategory:为类别标签
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #6个文档向量
    numWords = len(trainMatrix[0]) #特征单词的数量：32
    pAbusive = sum(trainCategory)/float(numTrainDocs) #辱骂文档的概率，类别标签为1的进行相加即可得到
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones(),32个单词的类条件概率,初始化为1，是为了拉普拉斯平滑
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #侮辱性言论
            p1Num += trainMatrix[i] #统计在侮辱性中该单词出现的概率
            p1Denom += sum(trainMatrix[i]) #在侮辱性中出现的单词总数
        else:  #非侮辱性言论
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)  #侮辱性言论中，该单词出现的概率 ,log避免许多小数相乘导致下溢      #change to log()
    p0Vect = log(p0Num/p0Denom)  #非侮辱性的言论中，该单词出现的概率       #change to log()
    return p0Vect,p1Vect,pAbusive #单词类条件概率，类概率

#vec2Classify为待分类向量,
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult,逐元素点乘,此处相加，其实相当于log里面概率相乘，即出现各个单词的概率相乘得到句子的概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
#词袋模型,对每个单词进行计数    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 #现在是计数加1，而不是简单的置1
    return returnVec

#测试朴素贝叶斯函数
def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses)) #得到各单词的类条件概率，类概率
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString) #分割符是任意除单词，数字之外的字符
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #过滤掉长度小于3的字符串，并返回list
#垃圾邮件测试    
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):#共有26个文件
        wordList = textParse(open('email/spam/%d.txt' % i).read()) #将垃圾邮件转换为词汇列表
        docList.append(wordList) #将整个列表当作元素添加到原列表中
        fullText.extend(wordList) #将列表中所有元素逐个添加到列表中，但并未去重复单词
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read()) #将非垃圾邮件转换为词汇列表
        docList.append(wordList)
        fullText.extend(wordList) #append将列表中每个元素都添加到fullText中
        classList.append(0) #最后一共有50个类标签
        ###
    vocabList = createVocabList(docList)#create vocabulary,创建词汇列表,去掉重复元素,共有692个元素
    trainingSet = range(50); testSet=[]           #create test set,创建测试列表
    for i in range(10):  #从中随机选出10个文本作为测试集，1/5
        randIndex = int(random.uniform(0,len(trainingSet))) #正态分布，从0~50
        testSet.append(trainingSet[randIndex]) #该文本索引加入测试集中
        del(trainingSet[randIndex])  #从训练集中删除该文本索引
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0, 从训练集中开始训练样本
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))#进行分类训练，计算单词条件概率以及类概率,array直接将list转化为ndarray
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:#进行分类，若分类有误，则记录下来
            errorCount += 1
            print "classification error",docList[docIndex] #分类错误，则需要进行打印原文档
    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText
#高频词去除函数，即一些停用词
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]    #只取 根据频率排名0~29的  
#http://newyork.craigslist.org/stp/index.rss 可以访问
def localWords(feed1,feed0):
    import feedparser #feedparser是一个RSS解析器，来源于 https://code.google.com/p/feedparser/
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary']) #每次访问一条rss源，并将其转化为词的列表,当作一个文档
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary,已经进行了去重操作
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words,去除出现频率最高的30个词语
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20): #选出其中的20个rss源进行测试
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i])) #如果条件概率较大，则作为元组加入到topSF中
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True) #lambda中，以第二个属性进行比较
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print item[0],item[1]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print item[0],item[1]

#test
list0Posts,listClasses= loadDataSet()
myVocabList=createVocabList(list0Posts)
print u"词汇表",myVocabList
vec1=setOfWords2Vec(myVocabList,list0Posts[0])
print vec1
vec2=setOfWords2Vec(myVocabList,list0Posts[3])
print vec2 
print u"######"
trainMat=[]
for postinDoc in list0Posts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
p0V,p1V,pAb=trainNB0(trainMat,listClasses)

print u"类条件概率p0V：",p0V
print u"灰条件概率p1V：",p1V
print u"类概率pAb：",pAb
print u"测试 testingNB():"
testingNB()
print u"spamTest:"
spamTest()

print u"rss测试"
import feedparser
ny=feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf=feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList,pSF,pNy=localWords(ny,sf)

#getTopWords
getTopWords(ny,sf)

