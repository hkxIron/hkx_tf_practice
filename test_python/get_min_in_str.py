# -*- coding:utf8 -*- 
#! /usr/bin/python
#coding:utf-8
import sys,math
from odps.udf import annotate
"""
1234,456 
"""
#udtf
@annotate('*->double')
class get_min_in_str(object):
    def _init_(self):
        pass
    def evaluate(self,num_str,sep=" ",default_min_num=0.0):
        sys.stderr.write("num_str"+str(num_str)+'\n')
        try:
            x=[float(ele) for ele in num_str.split(sep)]
        except Exception,e:
            sys.stderr.write("convert to float error!"+str(e))
            return default_min_num
        min_num=default_min_num;
        if len(x)>0:
            min_num=x[0]
            for ele in x:
                if ele<min_num:
                    min_num=ele
        return min_num