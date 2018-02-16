# -*- coding:utf8 -*- 
#! /usr/bin/python
#coding:utf-8
import sys,math
from odps.udf import annotate
from odps.udf import BaseUDTF
"""
输入文档id,向量的列, 即d1_d2:1,2,3, 4,5,6
输出文档id，欧氏距离，cos相似度
"""
#udtf
@annotate('*->string,double,double')
class getEuclideanAndCosDistance(object):
    def _init_(self):
        pass
    def process(self,doc_id,*vec):
        sys.stderr.write('doc_id:'+doc_id+'\n')
        sys.stderr.write("two vec size"+str(len(vec)))
        if len(vec)%2!=0: return #向量长必须为偶数
        vec_size=len(vec)/2
        x=vec[0:vec_size]
        y=vec[vec_size:]
        sum_eucli=2**30
        sum_x=0.0;sum_y=0.0;sum_xy=0.0
        for i in xrange(vec_size):
            sum_eucli+=(x[i]-y[i])**2
            sum_x+=x[i]**2
            sum_y+=y[i]**2
            sum_xy+=x[i]*y[i]
        sum_eucli=math.sqrt(sum_eucli)    
        div=sum_x**0.5*sum_y**0.5
        cos_similar=0
        if div>0:
            cos_similar=abs(sum_xy*1.0/div)
        sys.stderr.write("doc_id:%s,eucli_dist:%f,cos_dist:%f"%(doc_id,sum_eucli,1-cos_similar))
        self.forward(doc_id,sum_eucli,1-cos_similar)