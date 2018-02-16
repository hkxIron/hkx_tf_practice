#-*- coding: UTF-8 -*-
#!/usr/bin/env python
import sys,time,datetime
from odps.udf import annotate
from odps.udf import BaseUDTF
"""
wm_concat('|',concat(time,':',cate_leaf_id,':',action_type,':',weight)) as cate_action_list
get_user_pv_action_list(user_id,cate_action_list) as (user_id,group_time,pv_cate_list,action_cate_list) 
""" 
@annotate("string,string->string,string,string,string")
class get_user_pv_action_list(BaseUDTF):
    ctrlA = chr(1)  
    def process(self, user_id, behavior_list_str):
        sys.stderr.write("user_id:"+user_id+" behavior_list_str:"+str(behavior_list_str)+"\n")
        clc_list = behavior_list_str.split('|')
        res = []
        cate_set_pv= set()
        cate_list_pv=[]
        
        cate_set_action=set()
        cate_list_action=[]
        
        sort_action_cate=[]
        for x in clc_list:
            ele = x.split(':') 
            if len(ele)<4:continue 
            cate=ele[1]
            try:
                actionTime=ele[0]
                actionType=int(ele[2]) #0为pv，大于0为action
                actionWeight=float(ele[3]) #
            except Exception,e:
                sys.stderr.write("convert to int or float error!Please check:\n"+str(e))
                continue
            sort_action_cate.append([actionTime,cate,actionType,actionWeight]) #类目，时间，类型，权重
        #按时间排序
        sort_action_cate=sorted(sort_action_cate,key=lambda x:x[0])
        sys.stderr.write("sorted list:"+str(sort_action_cate)+"\n")
        last_pv_cate=-1;last_action_cate=-1;last_action_weight=-1;
        last_second=0;cur_second=0;
        SECOND_GAP=30*60 #大于半个小时的为一个gap,其边续行为必须在一半个小时以内
        time_item_group=[] #[[pvList,actionList],[pvList,actionList]],pvList
        pvGroup=[];actionGroup=[];
        for index in xrange(len(sort_action_cate)):
            if len(sort_action_cate[index])!=4:continue
            actionTime=sort_action_cate[index][0]
            cate=sort_action_cate[index][1]
            actionType=sort_action_cate[index][2]
            actionWeight=sort_action_cate[index][3]
            try:
                cur_second=int(time.mktime(time.strptime(actionTime,"%Y%m%d%H%M%S"))) #20160412183131
            except Exception,e:cur_second=0;
            if index==0:
                if actionType==0:
                    pvGroup=[[actionTime,cate]]
                    last_pv_cate=cate
                else:
                    actionGroup=[[actionTime,cate,actionWeight]]
                    last_action_cate=cate
                    last_action_weight=actionWeight
            elif index>=1:
                sys.stderr.write("cur_second:"+str(cur_second)+" last_second:"+str(last_second)+"\n")
                if abs(cur_second-last_second)>SECOND_GAP:#大于时间间隔
                    sys.stderr.write("gap is larger:"+str(cur_second-last_second)+"\n")
                    time_item_group.append([pvGroup,actionGroup])
                    if actionType==0:#pv
                        pvGroup=[[actionTime,cate]]
                        actionGroup=[]
                        last_pv_cate=cate
                    else:#action
                        pvGroup=[]
                        actionGroup=[[actionTime,cate,actionWeight]]
                        last_action_cate=cate
                        last_action_weight=actionWeight
                else:#小于时间间隔
                    if actionType==0:#pv
                        if last_action_cate!=cate:
                            pvGroup.append([actionTime,cate])
                            last_pv_cate=cate
                    else:#action
                        if last_action_cate!=cate or last_action_weight!=actionType:
                            actionGroup.append([actionTime,cate,actionWeight])
                            last_action_cate=cate
                            last_action_weight=actionWeight
            #更新时间
            last_second=cur_second
        #end of for
        #将最后剩下的加入到此表中
        time_item_group.append([pvGroup,actionGroup])#[[pvList,actionList],[pvList,actionList]], pvList:[[time,cate],[time,cate]],actionList:[[time,cate,weight],[time,cate,weight]]
        sys.stderr.write("time_item_group:"+str(time_item_group)+"\n")
        #----------------------------------------------
        #[[pv_set,action_dict],[pv_set,action_dict]],pv_set:{}
        if len(time_item_group) > 1:
            for group in time_item_group:#对于每个session里的行为
                if len(group)!=2:continue#group 里至少有一个商品才对
                pvGroup=group[0]
                actionGroup=group[1]
                early_time="30160412183131";#一千年以后
                if len(pvGroup)!=0:
                    early_time=pvGroup[0][0]
                if len(actionGroup)!=0 and actionGroup[0][0]<early_time:
                    early_time=actionGroup[0][0]
                #对于actionGroup里的每个行为类目，一定要在在pvGroup中
                pvSet=set()
                actionDict=dict()
                for pvCate in pvGroup:
                    if len(pvCate)!=2:continue
                    pvSet.add(pvCate[1]) #加入集合
                #    
                for cur_action in actionGroup:
                    if len(cur_action)!=3:continue #[time,cate,weight]
                    if cur_action[1] not in pvSet:continue
                    if cur_action[1] in actionDict.keys():
                        actionDict[cur_action[1]]=max(actionDict[cur_action[1]],cur_action[2]) #权重取大值
                    else:
                        actionDict[cur_action[1]]=cur_action[2]
                        
                pv_str=" ".join(list([str(x) for x in pvSet]))
                action_str=" ".join(list([str(k)+":"+str(v) for (k,v) in actionDict.items()]))
                pv_str=pv_str.strip()    
                action_str=action_str.strip() 
                if len(pv_str)>0 and len(action_str)>0:
                    self.forward(user_id,early_time,pv_str,action_str)
                #self.forward(user_id,early_time, ' '.join(list([x[1] for x in pvGroup])),' '.join(list([x[1]+":"+str(x[2]) for x in actionGroup]))) #group:[[t1,i1],[t2,i2]]