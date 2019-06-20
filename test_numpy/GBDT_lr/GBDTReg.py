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

    def caculate_split_loss(self, train_data, residual_gradient, cur_dim, split_value):
        """
        计算某种切分下的loss
        :param train_data: [N*dim]
        :param cur_dim: 当前切分的维度
        :param split_value: 当前维度的切分值
        :return: 切分loss
        """
        leftSubTree=[]
        rightSubTree=[]
        # 小于给左子树,大于给右子树
        for i in range(len(train_data)):
            sample_value=train_data[i, cur_dim]
            if sample_value<=split_value:
                leftSubTree.append(residual_gradient[i])
            else:
                rightSubTree.append(residual_gradient[i])
        # 分别计算左右子树的loss
        sumLoss=0.0
        #TODO:注意: 对于分类loss，在分裂结点时，叶子结点的值应该用newton-Raphson估计，而不是平均法
        sumLoss+=self.calculateSquareLoss(np.array(leftSubTree))
        sumLoss+=self.calculateSquareLoss(np.array(rightSubTree))
        return sumLoss

    def splitTree(self,x_train, residualGradient, treeHeight):
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
        # 计算如果不分裂时的整体loss，(xi-mean)^2
        cur_loss = self.calculateSquareLoss(residualGradient)
        minLossValue=cur_loss

        if treeHeight==self.maxTreeLength: return cur_loss

        # 以最优的分裂维度以及分裂点的值作为key -> (左子树,右子树)
        tree=dict([])

        # 寻找最优切分点:遍历每个特征维度下的所有样本
        for cur_dim in range(feature_dim):
            for best_split_index in range(data_size):
                # 实际在使用中,一般会先对特征值大小对样本进行排序
                split_value = x_train[best_split_index, cur_dim]
                leftSubTree=[]
                rightSubTree=[]
                # 假设以cur_dim作为最优切分特征维，best_split_index作为最优切分数据下标
                # 计算如此切分时的loss
                splitLoss = self.caculate_split_loss(x_train, residualGradient, cur_dim, split_value)
                if splitLoss<minLossValue:
                    bestSplitPointDim=cur_dim
                    bestSplitPointValue=split_value
                    minLossValue=splitLoss
        # 如果损失值没有变小，则不作任何改变，即并不需要分裂
        if minLossValue==cur_loss: return cur_loss
            #return np.mean(residualGradient)
        else:
            # 将上一次的残差梯度直接分配过去,并没有重新计算,即同一颗树里的残差并不需要再次计算
            leftSplitSample=[(x_train[i], residualGradient[i]) for i in range(data_size) if x_train[i, bestSplitPointDim] <= bestSplitPointValue]
            rightSplitSample=[(x_train[i], residualGradient[i]) for i in range(data_size) if x_train[i, bestSplitPointDim] > bestSplitPointValue]

            newLeftTreeSample = list(zip(*leftSplitSample))[0]
            newLeftSampleResidual = list(zip(*leftSplitSample))[1]
            newRightTreeSample = list(zip(*rightSplitSample))[0]
            newRightSampleResidual = list(zip(*rightSplitSample))[1]

            # 递归分裂左子树
            leftTree = self.splitTree(np.array(newLeftTreeSample), newLeftSampleResidual, treeHeight + 1)
            # 递归分裂右子树
            rightTree = self.splitTree(np.array(newRightTreeSample), newRightSampleResidual, treeHeight + 1)
            # 以最优的分裂维度以及分裂点的值作为key进行分裂左右子树
            tree[(bestSplitPointDim, bestSplitPointValue)]=[leftTree, rightTree]
            return tree

    #计算树的叶子节点数
    def getTreeLeafNodeNum(self,tree):
            size=0
            if type(tree) is not dict:
                return 1
            for item in tree.items():
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

        self.xValue=0 # 该样本落入的叶子结点的值
        xFeature=[0]*treeLeafNodeNum # x向量,
        self.leftZeroNum=0

        def scan(curTree, singleX):
            # 遍历当前树的所有结点
            for item in curTree.items():
                splitDim, splitValue=item[0]
                subLeftTree, subRightTree=item[1]
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
        xFeature[self.leftZeroNum]=1 # 生成特征时,看左边有多少个非0特征
        return self.xValue, xFeature

    #sigmoid函数
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-1*x))

    #建立GBDT树
    def buildGbdt(self,x_train,y_train):
        #size = len(x_train)
        #feature_dim = len(x_train[0])
        size, feature_dim = x_train.shape
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        x_train_feature=[]

        #初始化第一棵树
        treePreviousValue=0*y_train
        treeValues=[]
        treeValues.append(treePreviousValue)

        curValue = self.sigmoid(0*y_train) # 当前的预测值
        """
        当前的cross-entropy-loss: L= -(yi*log(pi)+(1-yi)*log(1-pi))
        loss对f(xi)的导数为:
        dL/df(xi) = pi -yi
        
        梯度更新值: -1*learning_rate* dL/df(xi), 即需要拟合的残差
        
        """
        dataFeatures=[]
        for i in range(self.maxTreeNum):
            print("the tree %i-th"%i)
            # 计算每个样本的梯度,只有当分裂一颗新树时，才需要计算残差并拟合
            residualGradient = -1*self.learningRate*(curValue-y_train)
            # 建立一颗树去拟合残差
            curTree = self.splitTree(x_train, residualGradient, treeHeight=1)
            self.tree.append(curTree)
            #print("tree index:",i, " tree:", curTree)
            curTreeLeafNodeNum = self.getTreeLeafNodeNum(curTree)
            curTreeValue=[]

            for singleX in x_train:
                xValue, _ = self.scanTree(curTree,singleX,curTreeLeafNodeNum)
                curTreeValue.append(xValue)

            treePreviousValue=np.array(curTreeValue)+treePreviousValue
            curValue=self.sigmoid(treePreviousValue)
            print(y_train)
            print("curValue:",curValue)

    #根据建成的树构建输入数据的特征向量
    def generateFeatures(self,x_train):
        dataFeatures=[]
        for curTree in self.tree:
            curFeatures=[]
            curTreeLeafNodeNum = self.getTreeLeafNodeNum(curTree)
            # print ("tree leaf node is %i"%(curTreeLeafNodeNum))
            # 为数据集中每个样本生成gbdt 的feature
            for singleX in x_train:
                _,xFeature = self.scanTree(curTree,singleX,curTreeLeafNodeNum)
                curFeatures.append(xFeature)

            if len(dataFeatures)==0:
                dataFeatures=np.array(curFeatures)
            else:
                dataFeatures=np.concatenate([dataFeatures,curFeatures],axis=1) # 将同一个样本,在不同树下的特征按列拼接起来
        return dataFeatures
