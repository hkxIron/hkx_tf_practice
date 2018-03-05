#coding=utf-8
__author__ = 'luchi.lc'
import numpy as np

"""
date:29/6/2017
usage:构造GBDT树并用其生成数据新的特征向量
"""
class GBDT(object):

    def __init__(self,config):

        self.learningRate = config.learningRate
        self.maxTreeLength=config.maxTreeLength
        self.maxLeafCount=config.maxLeafCount
        self.maxTreeNum=config.maxTreeNum
        self.tree=[]

    #计算平方损失
    def calculateSquareLoss(self,residual):
        """
        :param residual:梯度残差值
        :return:总体的残差值
        """
        mean = np.mean(residual)
        sumError = np.sum([(value-mean)**2 for value in residual])
        return sumError

    def caculate_split_loss(self,train_data,residual_gradient,cur_dim,split_value):
        """
        计算某种切分下的loss
        :param train_data: [N*dim]
        :param cur_dim:
        :return: 切分loss
        """
        leftSubTree=[]
        rightSubTree=[]
        for k in range(len(train_data)):
            tmpNum=train_data[k, cur_dim]
            if tmpNum<=split_value:
                leftSubTree.append(residual_gradient[k])
            else:
                rightSubTree.append(residual_gradient[k])
        # 分别计算左右子树的loss
        sumLoss=0.0
        sumLoss+=self.calculateSquareLoss(np.array(leftSubTree))
        sumLoss+=self.calculateSquareLoss(np.array(rightSubTree))
        return sumLoss

    def splitTree(self,x_train,residualGradient,treeHeight):
        """
        :param x_train:训练数据
        :param residualGradient:当前需要拟合的梯度残差值
        :param treeHeight 树的高度
        :return 建好的GBDT树
        """
        data_size = len(x_train)
        feature_dim = len(x_train[0])
        # 约定：左子树是小于等于，右子树是大于
        bestSplitPointDim=-1
        bestSplitPointValue=-1
        curLoss = self.calculateSquareLoss(residualGradient)
        minLossValue=curLoss
        if treeHeight==self.maxTreeLength:
            return curLoss
        tree=dict([])
        # 寻找最优切分点:遍历每个特征维度下的所有样本
        for cur_dim in range(feature_dim):
            for best_split_index in range(data_size):
                split_value = x_train[best_split_index, cur_dim]
                leftSubTree=[]
                rightSubTree=[]
                # 假设以cur_dim作为最优切分特征维，best_split_index作为最优切分数据下标
                # 计算如此切分时的loss
                sumLoss = self.caculate_split_loss(x_train,residualGradient,cur_dim,split_value)
                if sumLoss<minLossValue:
                    bestSplitPointDim=cur_dim
                    bestSplitPointValue=split_value
                    minLossValue=sumLoss
        # 如果损失值没有变小，则不作任何改变，也就是下面的归位一个Node
        if minLossValue==curLoss:
            return np.mean(residualGradient)
        else:
            leftSplit=[(x_train[i],residualGradient[i]) for i in range(data_size) if x_train[i, bestSplitPointDim] <= bestSplitPointValue]
            rightSplit=[(x_train[i],residualGradient[i]) for i in range(data_size) if x_train[i, bestSplitPointDim] > bestSplitPointValue]

            newLeftSubTree = zip(*leftSplit)[0]
            newLeftResidual = zip(*leftSplit)[1]
            # 递归分裂左子树
            leftTree = self.splitTree(np.array(newLeftSubTree),newLeftResidual,treeHeight+1)

            newRightSubTree = zip(*rightSplit)[0]
            newRightResidual =zip(*rightSplit)[1]
            # 递归分裂右子树
            rightTree = self.splitTree(np.array(newRightSubTree),newRightResidual,treeHeight+1)

            tree[(bestSplitPointDim,bestSplitPointValue)]=[leftTree,rightTree]
            return tree

    #计算树的节点数
    def getTreeLeafNodeNum(self,tree):
            size=0
            if type(tree) is not dict:
                return 1
            for item in tree.iteritems():
                subLeftTree,subRightTree=item[1]
                if type(subLeftTree) is dict:
                    size+=self.getTreeLeafNodeNum(subLeftTree)
                else:
                    size+=1

                if type(subRightTree) is dict:
                    size+=self.getTreeLeafNodeNum(subRightTree)
                else:
                    size+=1
            return size

    #遍历数据应该归到那个叶子节点，并计算其左侧的叶子节点个数
    def scanTree(self,curTree,singleX,treeLeafNodeNum):
        """

        :param curTree:当前的树
        :param singleX:需要送入到决策树的数据
        :param treeLeafNodeNum:树的叶子结点个数
        :return:该数据应该分到的叶子结点的值和其在当前树的转换的特征向量
        """

        self.xValue=0
        xFeature=[0]*treeLeafNodeNum
        self.leftZeroNum=0
        def scan(curTree,singleX):

            for item in curTree.iteritems():
                splitDim,splitValue=item[0]
                subLeftTree,subRightTree=item[1]
                if singleX[splitDim]<=splitValue:
                    if type(subLeftTree) is dict:
                        scan(subLeftTree,singleX)
                    else:
                        self.xValue=subLeftTree
                        return
                else:
                    self.leftZeroNum+=self.getTreeLeafNodeNum(subLeftTree)
                    if type(subRightTree) is dict:
                        scan(subRightTree,singleX)
                    else:
                        self.xValue=subRightTree
                        return
        scan(curTree,singleX)
        xFeature[self.leftZeroNum]=1
        return self.xValue,xFeature

    #sigmoid函数
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-1*x))
    #建立GBDT树
    def buildGbdt(self,x_train,y_train):
        size = len(x_train)
        feature_dim = len(x_train[0])
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        x_train_feature=[]

        #初始化第一棵树
        treePreviousValue=0*y_train
        treeValues=[]
        treeValues.append(treePreviousValue)

        curValue = self.sigmoid(0*y_train)
        dataFeatures=[]
        for i in range(self.maxTreeNum):
            print("the tree %i-th"%i)
            # 计算每个样本的梯度
            residualGradient = -1*self.learningRate*(curValue-y_train)
            # 建立一颗树
            curTree = self.splitTree(x_train,residualGradient,1)
            self.tree.append(curTree)
            print(curTree)
            curTreeLeafNodeNum = self.getTreeLeafNodeNum(curTree)
            curTreeValue=[]
            for singleX in x_train:
                xValue,xFeature = self.scanTree(curTree,singleX,curTreeLeafNodeNum)
                curTreeValue.append(xValue)

            treePreviousValue=np.array(curTreeValue)+treePreviousValue
            curValue=self.sigmoid(treePreviousValue)
            print(y_train)
            print("curValue")
            print(curValue)

    #根据建成的树构建输入数据的特征向量
    def generateFeatures(self,x_train):
        dataFeatures=[]
        for curTree in self.tree:
            curFeatures=[]
            curTreeLeafNodeNum = self.getTreeLeafNodeNum(curTree)
            # print ("tree leaf node is %i"%(curTreeLeafNodeNum))
            for singleX in x_train:
                _,xFeature = self.scanTree(curTree,singleX,curTreeLeafNodeNum)
                curFeatures.append(xFeature)

            if len(dataFeatures)==0:
                dataFeatures=np.array(curFeatures)
            else:
                dataFeatures=np.concatenate([dataFeatures,curFeatures],axis=1)

        return dataFeatures
