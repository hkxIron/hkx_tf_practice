# -*- coding: utf-8 -*-  
from matplotlib import pyplot  
import scipy as sp  
import numpy as np  
from matplotlib import pylab  
from sklearn.datasets import load_files  
from sklearn.cross_validation import train_test_split  
from sklearn.feature_extraction.text import  CountVectorizer  
from sklearn.feature_extraction.text import  TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report  
from sklearn.linear_model import LogisticRegression  
import time  
  
start_time = time.time()  
#绘制R/P曲线  
def plot_pr(auc_score, precision, recall, label=None):  
    pylab.figure(num=None, figsize=(6, 5))  
    pylab.xlim([0.0, 1.0])  
    pylab.ylim([0.0, 1.0])  
    pylab.xlabel('Recall')  
    pylab.ylabel('Precision')  
    pylab.title('P/R (AUC=%0.2f) / %s' % (auc_score, label))  
    pylab.fill_between(recall, precision, alpha=0.5)  
    pylab.grid(True, linestyle='-', color='0.75')  
    pylab.plot(recall, precision, lw=1)      
    pylab.show()  
  
#读取  
folder_prefix="../data/review_polarity/tokens/"

movie_data   = sp.load(folder_prefix+'movie_data.npy')  
movie_target = sp.load(folder_prefix+'movie_target.npy')  
x = movie_data  
y = movie_target  
  
#BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口  
count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english')  #tfidf  
average = 0  
testNum = 10  

for i in range(0, testNum):  
    #加载数据集，切分数据集80%训练，20%测试  
    x_train, x_test, y_train, y_test= train_test_split(movie_data, movie_target, test_size = 0.2)  
    x_train = count_vec.fit_transform(x_train)  
    x_test  = count_vec.transform(x_test)  
  
    #训练LR分类器  
    clf = LogisticRegression()  
    clf.fit(x_train, y_train)  
    y_pred = clf.predict(x_test)  
    p = np.mean(y_pred == y_test) 
    print "第%d次切分,准确率为:%.4f"%(i+1,p) 
    #print(p)  
    average += p  
  
      
#准确率与召回率  
answer = clf.predict_proba(x_test)[:,1]  #clf.predict_proba(x_test)维度:400*2,第一列是预测为0的概率，第二列是预测为1的概率
precision, recall, thresholds = precision_recall_curve(y_test, answer)      
report = answer > 0.5  
print(classification_report(y_test, report, target_names = ['neg', 'pos']))  
print("average precision:", average/testNum)  
print("time spent(s):", time.time() - start_time)  
  
plot_pr(0.5, precision, recall, "pos")
  