# -*- coding:utf8 -*- 
#! /usr/bin/python
#coding:utf-8
import sys,math
from odps.udf import annotate
import dis
"""
输入文档id,向量的列, 即d1_d2:1,2,3, 4,5,6,即doc_x,doc_y,vec_x,vec_y,sep="|"
输出文档id，欧氏距离，cos相似度，即str(doc_x),str(doc_y),sum_eucli,1-cos_similar
"""
#udtf
@annotate('*->double')
class get_euclid_dist(object):
    def _init_(self):
        pass
    def evaluate(self,vec_x,vec_y,sep="|"):
        dist=2**30
        sys.stderr.write(" vec_x:"+str(vec_x)+'\n')
        sys.stderr.write(" vec_y:"+str(vec_y)+'\n')
        try:
            x=[float(ele) for ele in vec_x.split(sep)]
            y=[float(ele) for ele in vec_y.split(sep)]
        except Exception,e:
            sys.stderr.write("convert to float error!"+str(e))
            return dist
        sys.stderr.write("vec_x size:"+str(len(x)))
        sys.stderr.write("vec_y size:"+str(len(y)))
        if len(x)!=len(y): 
            sys.stderr.write("input two vector size not equal")
            return dist
        if len(x)==0 or len(y)==0:return
        sum_of_diff=0.0
        for i in xrange(len(x)):
            sum_of_diff+=(x[i]-y[i])**2
        dist=sum_of_diff**0.5
        return dist