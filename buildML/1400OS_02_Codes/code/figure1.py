#coding:utf-8
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
#用sklearn中的load_iris读取数据
data = load_iris()
features = data['data']  #第3维是花瓣长度, 150*4
feature_names = data['feature_names'] #'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)',sepal:萼片，petal:花瓣
target = data['target']

#画出每两个维度之间的关系
pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
for i,(p0,p1) in enumerate(pairs): #pairs有6对
    plt.subplot(2,3,i+1)
    for t,marker,c in zip(range(3),">ox","rgb"):
        plt.scatter(features[target == t,p0], features[target == t,p1], marker=marker, c=c)
    plt.xlabel(feature_names[p0])
    plt.ylabel(feature_names[p1])
    plt.xticks([])
    plt.yticks([])
"""
target_names=["Virginica","Versicolor","Setosa"]
for leg,marker,c in zip(target_names,">ox","rgb"):
    plt.legend(["%s"%leg],loc="upper left")
"""
plt.savefig('../1400_02_01.png')

