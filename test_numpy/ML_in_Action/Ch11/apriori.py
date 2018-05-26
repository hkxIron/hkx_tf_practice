#coding:gbk
'''
Created on Mar 24, 2011
Ch 11 code
@author: Peter
'''
from numpy import *

def loadDataSet():
    return [[1, 3, 4],
            [2, 3, 5], 
            [1, 2, 3, 5], 
            [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return map(frozenset, C1)#use frozen set so we
                            #can use it as a key in a dict    

def scanD(D, Ck, minSupport):#D为事务列表集合，Ck为项列表,
    ssCnt = {} #字典
    for tid in D: #对每项事务
        for can in Ck:#对于每个单项
            if can.issubset(tid):#如果这个单项被该事务包含
                if not ssCnt.has_key(can): ssCnt[can]=1 #该单项没有被计数，进行计数
                else: ssCnt[can] += 1
               
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:#生成L1频繁项集
        support = ssCnt[key]/numItems #计算单项的支持度
        if support >= minSupport:
            retList.insert(0,key) #生成频繁1项集
            if support >=1:
                support=support
        supportData[key] = support #记录支持度,但不代表它是频繁的
    return retList, supportData
#已有频繁L(k-1)项集，产生候选k项集
def aprioriGen(Lk, k): #creates Ck,生成候选k项集,作者实现的方法并没有利用先验知识进行剪枝
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): #j为i的后一项
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()#生成候选k项集，前k-2项肯定要相同
            if L1==L2: #if first k-2 elements are equal
                #kexin Hu加入Apriori中的剪枝步骤
                un=Lk[i]|Lk[j]
                flag=True #假设为频繁
                unList=list(un) #得到un的副本list
                for ri in range(k):
                    item=unList.pop(ri) #依次检测k-1项集是否频繁
                    if frozenset(unList) not in Lk:#保证其每个子集必须是频繁的
                        flag=False
                        break
                    unList.insert(ri,item)  #恢复原来的list
                if flag and un not in retList:
                    retList.append(un) #set union,示并集，如{1,2}U{1,3}=>{1,2,3},因为它们前1项相同
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)#1候选项集
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):#L[k-2]:其实为刚刚添加进去的项集
        Ck = aprioriGen(L[k-2], k) #生成候选k项集
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk,扫描数事务集，去除不合要求的候选项集，得到频繁k项集
        supportData.update(supK) #更新支持度计数，其实是添加
        L.append(Lk) #lk项集添加到原列表
        k += 1
    return L, supportData
#生成关联规则,L为频繁项集
def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items,从至少含有两项的项集开始
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet] #得到Lk项集中的每一项,H作为规则后件
            if (i > 1):#Lk中若至少有3项
#                 if set(freqSet)==set([2,3,5]):
#                     print 'x'
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:#Lk中若只有两项，计算置信度
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

def calcConf(freqSet, H, supportData, brl, minConf=0.7):#计算置信度
    prunedH = [] #create new list to return
    for conseq in H: #H总是作为规则后件
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence，计算置信度
#         if set(freqSet)==set([2,3,5]):
#             print 'x'
        if conf >= minConf: 
            print freqSet-conseq,'-->',conseq,'conf:',conf
            brl.append((freqSet-conseq, conseq, conf)) #记录规则
            prunedH.append(conseq) #记录后件
    return prunedH #返回后件列表

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0]) #得到规则后件中元素个数
    if (len(freqSet) > (m + 1)): #try further merging,若频繁项集中元素比规则后件个数多2，则进行合并
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates,由频繁m项集，生成候选m+1项集
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
def pntRules(ruleList, itemMeaning):
    for ruleTup in ruleList:
        for item in ruleTup[0]:
            print itemMeaning[item]
        print "           -------->"
        for item in ruleTup[1]:
            print itemMeaning[item]
        print "confidence: %f" % ruleTup[2]
        print       #print a blank line

from time import sleep        
'''            
from votesmart import votesmart
votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#votesmart.apikey = 'get your api key first'
def getActionIds():
    actionIdList = []; billTitleList = []
    fr = open('recent20bills.txt') 
    for line in fr.readlines():
        billNum = int(line.split('\t')[0])
        try:
            billDetail = votesmart.votes.getBill(billNum) #api call
            for action in billDetail.actions:
                if action.level == 'House' and \
                (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
                    actionId = int(action.actionId)
                    print 'bill: %d has actionId: %d' % (billNum, actionId)
                    actionIdList.append(actionId)
                    billTitleList.append(line.strip().split('\t')[1])
        except:
            print "problem getting bill %d" % billNum
        sleep(1)                                      #delay to be polite
    return actionIdList, billTitleList

                
def getTransList(actionIdList, billTitleList): #this will return a list of lists containing ints
    itemMeaning = ['Republican', 'Democratic']#list of what each item stands for
    for billTitle in billTitleList:#fill up itemMeaning list
        itemMeaning.append('%s -- Nay' % billTitle)
        itemMeaning.append('%s -- Yea' % billTitle)
    transDict = {}#list of items in each transaction (politician) 
    voteCount = 2
    for actionId in actionIdList:
        sleep(3)
        print 'getting votes for actionId: %d' % actionId
        try:
            voteList = votesmart.votes.getBillActionVotes(actionId)
            for vote in voteList:
                if not transDict.has_key(vote.candidateName): 
                    transDict[vote.candidateName] = []
                    if vote.officeParties == 'Democratic':
                        transDict[vote.candidateName].append(1)
                    elif vote.officeParties == 'Republican':
                        transDict[vote.candidateName].append(0)
                if vote.action == 'Nay':
                    transDict[vote.candidateName].append(voteCount)
                elif vote.action == 'Yea':
                    transDict[vote.candidateName].append(voteCount + 1)
        except: 
            print "problem getting actionId: %d" % actionId
        voteCount += 2
    return transDict, itemMeaning
'''

if __name__=='__main__':
    print u"测试Apriori算法:"
    dataSet=loadDataSet()
    C1=createC1(dataSet)
    print u"C1\n",C1
    '''
    D=map(set,dataSet)
    L1,suppData0=scanD(D,C1,0.5)
    print u"L1\n",L1,u"\nsuppData0\n",suppData0
    '''
    L,suppData=apriori(dataSet,0.5)
    print u"\nL:\n",L,"\nsuppData:\n",suppData
    
    print u"生成关联规则："
    rules=generateRules(L,suppData,minConf=0.5)
    print u"rules:\n",rules
    
    print u"毒蘑菇测试..."
    mushDatSet=[line.split() for line in open('mushroom.dat').readlines()]
    L,suppData=apriori(mushDatSet,minSupport=0.4)
    for item in L[3]:
        if item.intersection('2'):#2代表有毒
            print item
            
    print u"结束"