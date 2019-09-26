# -*- coding: utf-8 -*-
"""
blog: http://www.csuldw.com/2016/03/12/2016-03-12-performance-evaluation/

Created on Sat Mar 12 17:43:48 2016

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plotROC1(predScore, labels):
    assert set(labels) == {0,1}
    numPos = np.sum(np.array(labels)==1)
    numNeg = len(labels)-numPos
    yStep = 1/np.float(numPos)  # y轴每步的步长
    xStep = 1/np.float(numNeg)
    sortedIndex = (-predScore).argsort() #对predScore进行降序排序，得到排序索引值
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    tpr_list = []
    fpr_list = []
    auc = 0.0
    last_x_change = 0.0
    for index in sortedIndex:
        threshold = predScore[index]
        pred_postive = predScore >= threshold
        tp = sum(np.logical_and(pred_postive, labels)) # 正例被识别成正例
        fp = sum(np.logical_and(pred_postive, 1-labels)) # 负例被识别成正例
        tpr = tp/numPos
        fpr = fp/numNeg # 1 - sp = FP/(TN+FP) = 1 - TN/(TN+FP)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        # 此时坐标(fpr,tpr)即为roc曲线中的点
        if labels[index] == 0: # 当前样本为负例,但被识别成正例
            auc += tpr* (fpr-last_x_change)
            # 或者auc += tpr*xStep
            last_x_change = fpr
    ax.plot(fpr_list, tpr_list, c='b') # 第一个是x,第二个是y
    ax.plot([0,1],[0,1],'b--') # 绘制对角线
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("auc: ", auc, " fpr:", fpr_list, " tpr:", tpr_list)

def plotROC2(predScore, labels):
    assert set(labels) == {0,1}
    numPos = np.sum(np.array(labels)==1)
    numNeg = len(labels)-numPos
    yStep = 1/np.float(numPos)  # y轴每步的步长
    xStep = 1/np.float(numNeg)
    sortedIndex = (-predScore).argsort() #对predScore进行降序排序，得到排序索引值
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    tpr_list = []
    fpr_list = []
    auc = 0.0
    last_x_change = 0.0
    tpr, fpr = 0.0, 0.0
    # 不再需要每次计算tp,fp
    for index in sortedIndex:
        # 此时坐标(fpr,tpr)即为roc曲线中的点
        if labels[index] == 0: # 当前为负例,但被识别成正例
            fpr += xStep
            auc += tpr* xStep
            # 或者auc += tpr*xStep
        else: # 当前为正例,且识别成正例
            tpr += yStep
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    ax.plot(fpr_list, tpr_list, c='b') # 第一个是x,第二个是y
    ax.plot([0,1],[0,1],'b--') # 绘制对角线
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("auc: ", auc, " fpr:", fpr_list, " tpr:", tpr_list)

# 没太看懂
def plotROC3(predScore, labels):
    point = (1, 1)
    ySum = 0.0
    assert set(labels) == {0,1}
    numPos = np.sum(np.array(labels)==1)
    numNeg = len(labels)-numPos
    yStep = 1/np.float(numPos)  # y轴每步的步长
    xStep = 1/np.float(numNeg)
    sortedIndex = predScore.argsort() #对predScore进行降序排序，得到排序索引值
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndex:
        # 此时坐标(fpr,tpr)即为roc曲线中的点
        # ------------
        if labels[index] == 1.0: #如果正样本各入加1，则x不走动，y往下走动一步
            delX = 0
            delY = yStep
        else:                   #否则，x往左走动一步，y不走动
            delX = xStep
            delY = 0
            ySum += point[1]     #统计y走动的所有步数的和
        ax.plot([point[0], point[0] - delX], [point[1], point[1] - delY],c='b')
        point = (point[0] - delX, point[1] - delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    ax.axis([0, 1, 0, 1])
    plt.show()
    #最后，所有将所有矩形的高度进行累加，最后乘以xStep得到的总面积，即为AUC值
    print("auc: ", ySum * xStep)

if __name__ == "__main__":
    np.random.seed(0)
    num = 10
    score = np.random.rand(num) # 0.916
    #score = np.random.rand(num)/2 # 0.916
    #score = np.zeros(num) # 0.5
    #score = np.zeros(num)+1 # 0.5
    label = np.random.randint(low=0,high=2,size=num)
    plotROC1(score, label)
    plotROC2(score, label)
    plotROC3(score, label)
    print('================')
    fpr, tpr, thresholds = metrics.roc_curve(label, score) # sklearn中的tpr,fpr只将x轴变动时的记录下来了
    auc = metrics.auc(fpr, tpr)
    print("sklearn auc:", auc," fpr:", fpr, " tpr:", tpr)
