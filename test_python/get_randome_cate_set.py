# -*- coding:utf8 -*- 
#! /usr/bin/python
#coding:utf-8
import sys,math
from odps.udf import annotate
import random
"""
"""
#udtf
@annotate('*->string')
class get_randome_cate_set(object):
    def _init_(self):
        pass
    def evaluate(self,cate_all,cate_action):
        if cate_all is None or cate_action is None: return None
        diff_set_str=""
        cate_all_set=set(cate_all.split(" "))
        #50014577:1 50008297:1
        cate_action_set=set([x.split(":")[0] for x in cate_action.split(" ")])
        diff_set=cate_all_set-cate_action_set
        neg_cate_size=min(len(diff_set),10*len(cate_action_set))
        #
        diff_set_list=random.shuffle(list(diff_set))#随机打散
        diff_set_list=diff_set_list[0:neg_cate_size]
        return " ".join(diff_set_list)