# spectral cluster
# blog:https://blog.csdn.net/hjimce/article/details/45749757

# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

# 生成两个高斯分布训练样本用于测试
# 构造第一类样本类
mean1 = [0, 0]
cov1 = [[1, 0], [0, 1]]  # 协方差矩阵
x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T

data = []
for x, y in zip(x1, y1):
    data.append([x, y])
# 构造第二类样本类
mean2 = [3, 3]
cov2 = [[1, 0], [0, 1]]  # 协方差矩阵
x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
for x, y in zip(x2, y2):
    data.append([x, y])
random.shuffle(data)  # 打乱数据
data = np.asarray(data, dtype=np.float32)

# 算法开始
# 计算两两样本之间的权重矩阵,在真正使用场景中，样本很多，可以只计算邻接顶点的权重矩阵
m, n = data.shape
print("m:", m, "n:", n)
distance = np.zeros((m, m), dtype=np.float32)

for i in range(m):
    for j in range(m):
        if i == j:
            continue
        dis = sum((data[i] - data[j]) ** 2)
        distance[i, j] = dis
# 构建归一化拉普拉斯矩阵
similarity = np.exp(-1. * distance / distance.std())
for i in range(m):
    similarity[i, i] = 0

for i in range(m):
    similarity[i] = -similarity[i] / sum(similarity[i])  # 归一化操作
    similarity[i, i] = 1  # 拉普拉斯矩阵的每一行和为0，对角线元素之为1

# 计算拉普拉斯矩阵的前k个最小特征值
[Q, V] = np.linalg.eig(similarity) # L = D - W, 求L的最小k个特征值对应的特征向量
idx = Q.argsort()
Q = Q[idx]
V = V[:, idx]
# 前3个最小特征值
num_clusters = 3
newd = V[:, :2] #这里聚成几类与取前几个特征向量之间没有关系

# 对降维后的特征向量进行k均值聚类
clf = KMeans(n_clusters=num_clusters)
clf.fit(newd)
# 显示结果
for i in range(data.shape[0]):
    if clf.labels_[i] == 0:
        plt.plot(data[i, 0], data[i, 1], 'go')
    elif clf.labels_[i] == 1:
        plt.plot(data[i, 0], data[i, 1], 'ro')
    elif clf.labels_[i] == 2:
        plt.plot(data[i, 0], data[i, 1], 'yo')
    elif clf.labels_[i] == 3:
        plt.plot(data[i, 0], data[i, 1], 'bo')

plt.show()
