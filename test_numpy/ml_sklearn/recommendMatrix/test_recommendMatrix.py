# -*- coding: utf-8 -*-  
from matplotlib import pyplot  
import scipy as sp  
import numpy as np  
from matplotlib import pylab  
from sklearn.datasets import load_files  
from sklearn.cross_validation import train_test_split  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report  
  
import time  
from scipy import sparse  
  
start_time = time.time()  
  
#计算向量test与data数据每一个向量的相关系数，data一行为一个向量  
def calc_relation(testfor, data):  
    return np.array(  
        [np.corrcoef(testfor, c)[0,1]  
         for c in data])  
  
# luispedro提供的加速函数:  
def all_correlations(y, X):  
    X = np.asanyarray(X, float)  
    y = np.asanyarray(y, float)  
    xy = np.dot(X, y)  
    y_ = y.mean()  
    ys_ = y.std()  
    x_ = X.mean(1)  
    xs_ = X.std(1)  
    n = float(len(y))  
    ys_ += 1e-5  # Handle zeros in ys  
    xs_ += 1e-5  # Handle zeros in x  
    return (xy - x_ * y_ * n) / n / xs_ / ys_  
  
          
#数据读入，数据的每行可表示为：某个用户买了某个商品多少次 
data = np.loadtxt('../data/recommendMatrix.txt')  #data:8行*3列
x_p = data[:, :2] # 取前2列  
y_p = data[:,  2] # 取第2列    
x_p -= 1          # 0为起始索引  
y = (sparse.csc_matrix((data[:,2], x_p.T)).astype(float))[:, :].todense() #todense()可以将稀疏矩阵转化为普通的矩阵 ,x3.toarrary(),x3.todense(),x3[:,:].todense()在此处输出是等价的
#csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
#where ``data``, ``row_ind`` and ``col_ind`` satisfy the
#relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
#矩阵y为2行4列
nUser, nItem = y.shape  
  
  
#可视化矩阵  
pyplot.imshow(y, interpolation='nearest')  
pyplot.xlabel('Item')  
pyplot.ylabel('User')  
pyplot.xticks(range(nItem))  
pyplot.yticks(range(nUser))  
pyplot.show()  
  
  
#加载数据集，切分数据集80%训练，20%测试  
x_p_train, x_p_test, y_p_train, y_p_test =train_test_split(data[:,:2], data[:,2], test_size = 0.0)      
x = (sparse.csc_matrix((y_p_train, x_p_train.T)).astype(float))[:, :].todense()  
      
  
Item_likeness = np.zeros((nItem, nItem))  
  
#训练      
for i in range(nItem):
    print "计算商品%d的相似商品及得分..."%i  
    Item_likeness[i] = calc_relation(x[:,i].T, x.T)  
    Item_likeness[i,i] = -1          
          
for t in range(Item_likeness.shape[1]):  
    item = Item_likeness[t].argsort()[-3:] #最后三个为概率最大的三个  
    print("Buy Item %d will buy item %d,%d,%d "%  
          (t, item[2], item[1], item[0]))  
  
print("time spent:", time.time() - start_time)  