# -*- coding: utf-8 -*-  
import numpy as np  
import scipy as sp  
from sklearn import svm  
from sklearn.cross_validation import train_test_split  
import matplotlib.pyplot as plt  
  
data   = []  
labels = []  
with open("../data/thin_fat.txt") as ifile:  
        for line in ifile:  
            tokens = line.strip().split(' ')  
            data.append([float(tk) for tk in tokens[:-1]])  #第一列与第二列是数据
            labels.append(tokens[-1])  #最后一列是标签

x = np.array(data)  #data是一个二维的列表,现在x为二维数组
labels = np.array(labels)  
y = np.zeros(labels.shape)  
y[labels=='fat']=1  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.0)  #全部都用来训练，没有测试用例
  
h = .02  # 步长  
# create a mesh to plot in  
x_min, x_max = x_train[:, 0].min() - 0.1, x_train[:, 0].max() + 0.1  
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1  
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  
                     np.arange(y_min, y_max, h))  
  
''''' SVM '''  
# title for the plots  
titles = ['LinearSVC (linear kernel)',  
          'SVC with polynomial (degree 3) kernel',  
          'SVC with RBF kernel',  
          'SVC with Sigmoid kernel']  
print "正在训练svm classfier..."
clf_linear  = svm.SVC(kernel='linear').fit(x, y)  
#clf_linear  = svm.LinearSVC().fit(x, y)  
clf_poly    = svm.SVC(kernel='poly', degree=3).fit(x, y) #3次多项式 
clf_rbf     = svm.SVC().fit(x, y)  
clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x, y)  
  
for i, clf in enumerate((clf_linear, clf_poly, clf_rbf, clf_sigmoid)):  #对元组进行迭代,i为索引
    answer = clf.predict(np.c_[xx.ravel(), yy.ravel()]) #xx.ravel()为按行展开 的向量，与flatten 相似,flatten会返回副本,np.c_在列向上堆叠,
    print(clf)  #打印训练器, xx.ravel.shape:80600,一维数组
    ##print(np.mean(answer == y_train))  #准确率,但是这样计算准确率是有问题的，二者维度不一样
    print(answer) #answer的维数为: 80600,y_train的维度为17
    print(y_train)  
  
    plt.subplot(2, 2, i + 1)  #指定使用第几个图像
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
      
    # Put the result into a color plot  
    z = answer.reshape(xx.shape)  
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)  #cmap是颜色映射表,如果去掉， 则不能显示分类的边界  
      
    # Plot also the training points  
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)  
    plt.xlabel(u'height')  
    plt.ylabel(u'weight')  
    plt.xlim(xx.min(), xx.max())  
    plt.ylim(yy.min(), yy.max())  
    #plt.xticks(())  #不要此行才能显示刻度
    #plt.yticks(())  
    plt.title(titles[i])  
      
plt.show()  
