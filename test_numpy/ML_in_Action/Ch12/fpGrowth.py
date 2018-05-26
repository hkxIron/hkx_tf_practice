#coding:gbk
'''
Created on Jun 14, 2011
FP-Growth FP means frequent pattern
the FP-Growth algorithm needs: 
1. FP-tree (class treeNode)
2. header table (use dict)

This finds frequent itemsets similar to apriori but does not 
find association rules.  

@author: Peter
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None #用于连接所有事务中的相同项
        self.parent = parentNode      #needs to be updated
        self.children = {} 
    
    def inc(self, numOccur):
        self.count += numOccur
        
    def disp(self, ind=1):
        print '|__'*(ind-1), self.name, ':', self.count
        for child in self.children.values():
            child.disp(ind+1)
#构建FP树,dataSet为frozenset 组成的dict
def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    headerTable = {}
    #go over dataSet twice
    for trans in dataSet:#first pass counts frequency of occurance，第一次遍历
        num=dataSet[trans]
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + num #进行+1,头指针亦即频繁1项集
    for k in headerTable.keys():  #remove items not meeting minSup
        if headerTable[k] < minSup: #移除支持度较小的项
            del(headerTable[k]) #删除字典中的一项
    freqItemSet = set(headerTable.keys())#频繁1项集
    #print 'freqItemSet: ',freqItemSet
    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out
    for k in headerTable:#头指针列表,亦即频繁1项集
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link ,'S':[3,None],None用来占位，指向树中相同的项
    #print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #create tree,创建空节点
    for tranSet, count in dataSet.items():  #go through dataset 2nd time,第二次遍历，键-值
        localD = {}#遍历事务集中的每一条事务
        for item in tranSet:  #put transaction items in order,如： frozenset(['e', 'm', 'q', 's', 't', 'y', 'x', 'z'])
            if item in freqItemSet:#如果item在频繁-1项集里
                localD[item] = headerTable[item][0] #得到频繁-1项集值
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]#按照项目出现在1项集中的次数进行降序排序
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset,
    return retTree, headerTable #return tree and header table

#items为该条事务记录中所有满足支持度的项目降序排序后的频繁1项集,inTree为Fp-growth树,（headerTable）头表：用来进行索引
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#如果该项目已经在Fp树中，那么对结点进行加1,check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children，
        inTree.children[items[0]] = treeNode(items[0], count, inTree) #将items[0]加入树孩子，即频繁1项集的首项
        if headerTable[items[0]][1] == None: #update header table #如果当前1项集还未指向树中一个项目
            headerTable[items[0]][1] = inTree.children[items[0]]#更新头指针列表,指向树中的相同频繁1项集
        else:#否则递归地更新头指针所指向的孩子
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#如果该条事务中还有其它项目，则对它们逐个建立节点,call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
        
def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
    
def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]#(sort header table)
    for basePat in bigL:  #start from bottom of header table
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print 'finalFrequent Item: ',newFreqSet    #append to set
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        #print 'condPattBases :',basePat, condPattBases
        #2. construct cond FP-tree from cond. pattern base
        myCondTree, myHead = createTree(condPattBases, minSup)
        #print 'head from conditional tree: ', myHead
        if myHead != None: #3. mine cond. FP-tree
            print 'conditional tree for: ',newFreqSet
            myCondTree.disp(1)            
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               #['z'],#重复Z
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):#创建事务字典
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1 #但会去掉重复值,retDict[frozenset(trans)]=retDict.get(frozenset(trans),0)+ 1
    return retDict
'''
import twitter
from time import sleep
import re

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = ''
    CONSUMER_SECRET = ''
    ACCESS_TOKEN_KEY = ''
    ACCESS_TOKEN_SECRET = ''
    api = twitter.Api(consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = createInitSet(parsedList)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList
'''

if __name__=='__main__':
    print u'测试 treeNode类:'
    rootNode=treeNode('pyramid',9,None)
    rootNode.children['eye']=treeNode("eye",13,None)
    rootNode.children['phoenix']=treeNode("phoenix",3,None)
    rootNode.disp()
    
    print u"测试FP树:"
    minSup = 3
    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFPtree.disp()
    
    print u"获取条件基："
    print findPrefixPath('x', myHeaderTab['x'][1])
    print findPrefixPath('z', myHeaderTab['z'][1])
    
    print u"挖掘频繁项集:" 
    myFreqList = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    print u"频繁项集：",myFreqList
    
    print u"从新闻网站点击流中挖掘:\n"
    parsedDat=[line.split() for line in open('kosarak.dat').readlines()] 
    initSet=createInitSet(parsedDat)
    myFPtree,myHeaderTab=createTree(initSet, 1e+5)
    myFreqList=[]
    mineTree(myFPtree, myHeaderTab, 1e+5, set([]), myFreqList)
    print len(myFreqList)
    print myFreqList
    