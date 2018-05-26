# -*- coding:utf8 -*- 
#! /usr/bin/python
#coding:utf-8
import sys,math
from odps.udf import annotate
"""
输入文档id,向量的列, 即d1_d2:1,2,3, 4,5,6,即doc_x,doc_y,vec_x,vec_y,sep="|"
输出文档id，欧氏距离，cos相似度，即str(doc_x),str(doc_y),sum_eucli,1-cos_similar
"""
#udtf
@annotate('*->double')
class get_cos_similar(object):
    def _init_(self):
        pass
    def evaluate(self,vec_x,vec_y,sep="|"):
        cos_similar=0
        sys.stderr.write(" vec_x:"+str(vec_x)+'\n')
        sys.stderr.write(" vec_y:"+str(vec_y)+'\n')
        try:
            x=[float(ele) for ele in vec_x.split(sep)]
            y=[float(ele) for ele in vec_y.split(sep)]
        except Exception,e:
            sys.stderr.write("convert to float error!"+str(e))
            return cos_similar
        sys.stderr.write("vec_x size:"+str(len(x)))
        sys.stderr.write("vec_y size:"+str(len(y)))
        if len(x)!=len(y): 
            sys.stderr.write("input two vector size not equal")
            return cos_similar
        if len(x)==0 or len(y)==0:return
        sum_x=0.0;sum_y=0.0;sum_xy=0.0
        for i in xrange(len(x)):
            sum_x+=x[i]**2
            sum_y+=y[i]**2
            sum_xy+=x[i]*y[i]
        div=sum_x**0.5*sum_y**0.5
        if div>0:
            cos_similar=sum_xy*1.0/div
        sys.stderr.write("cos_similar:%.4f"%(cos_similar))
        return cos_similar