# -*- coding:utf8 -*- 
#! /usr/bin/python
#coding:utf-8
class hash_str_to_int32(object):
    def evaluate(self,wordStr):
        seed = 31 # 31 131 1313 13131 131313 etc..
        hash = 0
        mod=2000000000 #2**61=2305843009213693952
        for i in range(len(wordStr)):
            hash = hash * seed + ord(wordStr[i])
            hash =  (hash & 0x7FFFFFFF)
            print "第%d次:hash value:%d"%(i,hash)
        #2**32=4294967296
        #hash =  (hash & 0x7FFFFFFF) %mod   
        #hash =  (hash) %mod
        return hash
    
ss="145200000201239003"
obj=hash_str_to_int32()
print "hash value:"+str(obj.evaluate(ss))