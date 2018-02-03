#!/usr/bin/env python
# encoding: utf-8
#
# 查找全量训练任务时非法的样本
# 返回不符合指定上限的特征以及其所在的样本

import os,sys,argparse

# _label__3269779:-1 111806 508261 4343061 4343692:0.1 4372633 4394483:0.2
def get_invalid_record(file_name,feature_name=None,count_limit=1):
    if not os.path.isfile(file_name):
        print("file %s doesn't exist!"%file_name)
        sys.exit(-1)
    with open(file_name,"r") as f:
        for line in f.readlines():
            feats=line.split(" ")[1:]
            feat_count={}
            for kv in feats:
                kv = kv.split(":")
                if len(kv)==1:
                    key=int(kv[0])
                    feat_count[key]=feat_count.get(key,0)+1
                elif len(kv)==2:
                    key=int(kv[0])
                    value=float(kv[1])
                    feat_count[key]=feat_count.get(key,0)+value
            if not feature_name:
                for (k,v) in feat_count.items():
                    if abs(v)>count_limit:
                       print("line:{} invalid key:{} value:{}".format(line,k,v))
            else:
                if abs(feat_count[feature_name])>count_limit:
                    print("line:{} invalid key:{} value:{}".format(line,feature_name,feat_count[feature_name]))

if __name__ =="__main__":
    if len(sys.argv)==1:
        print("Usage:%s --file_name [--feature_name] [--count_limit]"%sys.argv[0])
        sys.exit(-1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='test.txt', type=str,help="待查找的文件")
    parser.add_argument('--feature_name', default=None, type=str,help="待查找的特征哈希值，如果为空，将查找所有的特征")
    parser.add_argument('--count_limit', default=2, type=float,help="非法特征值的上限")
    FLAGS, unparsed = parser.parse_known_args()
    get_invalid_record(FLAGS.file_name,FLAGS.feature_name,FLAGS.count_limit)
