# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:50:11 2012

@author: wilrich
"""

import scipy as sp


def tfidf(term, doc, dataset):
    """
    注意:tfidf是与词和文档相关的,不存在脱离文档单独说某个词的tfidf
    :param term:
    :param doc:
    :param dataset:
    :return:
    """
    # 计算每个单词出现的频数
    count_of_each_term = [doc.count(w) for w in set(doc)]
    # 当前词的词频
    tf = float(doc.count(term))/sum(count_of_each_term)
    # 逆文档词频=文档总数/出现该词的文档数
    doc_list_has_term = [doc for doc in dataset if term in doc]
    idf = sp.log(float(len(dataset)) / (len(doc_list_has_term)))
    return tf * idf


a, abb, abc = ["a"], ["a", "b", "b"], ["a", "b", "c"]
D = [a, abb, abc]

print("tfidf:")
print(tfidf("a", a, D)) # a在每个doc里都有,且在a只含"a",所以tfidf=0
print(tfidf("b", abb, D)) # b夺abb中出现2次,所以tfidf较小
print(tfidf("a", abc, D)) # abc中出山现a一次,所以tfidf
print(tfidf("b", abc, D)) #
print(tfidf("c", abc, D)) # c只在abc中出现,所以较大
"""
0.0
0.27031007207210955
0.0
0.13515503603605478
0.3662040962227032
"""
