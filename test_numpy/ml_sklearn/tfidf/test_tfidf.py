# -*- coding: utf-8 -*-  
import scipy as sp  
import numpy as np  
from sklearn.datasets import load_files  
from sklearn.cross_validation import train_test_split  
from sklearn.feature_extraction.text import  TfidfVectorizer  
  
'''''加载数据集，切分数据集80%训练，20%测试'''  
movie_reviews = load_files('../data/endata')    
doc_terms_train, doc_terms_test, y_train, y_test\
    = train_test_split(movie_reviews.data, movie_reviews.target, test_size = 0.3)  
      
'''''BOOL型特征下的向量空间模型，注意，测试样本调用的是transform接口'''  
count_vec = TfidfVectorizer(binary = False, decode_error = 'ignore',stop_words = 'english')  
x_train = count_vec.fit_transform(doc_terms_train)  
x_test  = count_vec.transform(doc_terms_test)  
x       = count_vec.transform(movie_reviews.data)  
y       = movie_reviews.target  
print(doc_terms_train)  #打印出文档
print(count_vec.get_feature_names())  #打印出特征单词
print(x_train.toarray())  #toarray()即可打印出稀疏矩阵,打印出文档-单词矩阵,  每一行为一个完整的文档
print(movie_reviews.target) 
