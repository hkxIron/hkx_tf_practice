# -*- coding: utf-8 -*-  
from matplotlib import pyplot  
import scipy as sp  
import numpy as np  
from sklearn.datasets import load_files  
from sklearn.cross_validation import train_test_split  
from sklearn.feature_extraction.text import  CountVectorizer  
from sklearn.feature_extraction.text import  TfidfVectorizer  
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  

folder_prefix="../data/review_polarity/tokens/"

'''
print "正在加载数据" 
movie_reviews = load_files(folder_prefix)  
#保存 
sp.save(folder_prefix+'movie_data.npy', movie_reviews.data) 
sp.save(folder_prefix+'movie_target.npy', movie_reviews.target) 
'''
  
#读取  
movie_data   = sp.load(folder_prefix+'movie_data.npy')  
movie_target = sp.load(folder_prefix+'movie_target.npy')  
x = movie_data  
y = movie_target  
  
#BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口  
count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english')  
  
#加载数据集，切分数据集80%训练，20%测试  
x_train, x_test, y_train, y_test= train_test_split(movie_data, movie_target, test_size = 0.2)  
x_train = count_vec.fit_transform(x_train)  
x_test  = count_vec.transform(x_test)  
  
  
#调用MultinomialNB分类器,多项分布，这个分类器以出现次数作为特征值，我们使用的TF-IDF也能符合这类分布。
clf = MultinomialNB().fit(x_train, y_train)  
doc_class_predicted = clf.predict(x_test)  
      
#print(doc_class_predicted)  
#print(y)  
print(np.mean(doc_class_predicted == y_test))  
  
#准确率与召回率  
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))  
answer = clf.predict_proba(x_test)[:,1]  
print "x_test:"
print x_test.toarray() #将稀疏矩阵打印出来
report = answer > 0.5  
print(classification_report(y_test, report, target_names = ['neg', 'pos']))  
