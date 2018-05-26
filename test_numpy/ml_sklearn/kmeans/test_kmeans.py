# -*- coding: utf-8 -*-  
from matplotlib import pyplot  
import scipy as sp  
import numpy as np  
from sklearn import svm  
import matplotlib.pyplot as plt  
from sklearn.cluster   import KMeans  
from scipy import sparse  
  
#数据读入  
data = np.loadtxt('../data/kmeans_test_data.txt')  
x_p = data[:, :2] # 取前2列  
y_p = data[:,  2] # 取前2列  
x = (sparse.csc_matrix((data[:,2], x_p.T)).astype(float))[:, :].todense()  
nUser = x.shape[0]  
  
#可视化矩阵  
pyplot.imshow(x, interpolation='nearest')  
pyplot.xlabel('user')  
pyplot.ylabel('user')  
pyplot.xticks(range(nUser))  
pyplot.yticks(range(nUser))  
pyplot.show()  
  
#使用默认的K-Means算法  
num_clusters = 2  
clf = KMeans(n_clusters=num_clusters,  n_init=1, verbose=1)  
clf.fit(x)  
print(clf.labels_)  
  
#指定用户0与用户5作为初始化聚类中心  
init = np.vstack([ x[0], x[5] ])  
clf = KMeans(n_clusters=2, init=init)  
clf.fit(x)  
print(clf.labels_)  