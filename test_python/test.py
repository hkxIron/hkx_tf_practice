#coding:utf-8       #设置python文件的编码为utf-8，这样就可以写入中文注释
import re
import time,datetime
#from datetime import datetime,timedelta
#import time

def get_index_by_value(value,list):
    if value is None or len(str(value))==0 or len(list)==0:
        return -1  #如果为null就是-1
    ind=0
    for x in list:
       if value<=x:
           return ind 
       else:ind+=1
    return ind #超出范围了
def foo(arg1,arg2="OK",*tupleArg,**dictArg):
    print "arg1=",arg1
    print "arg2=",arg2
    for i,element in enumerate(tupleArg):
        print "tupleArg %d-->%s" % (i,str(element))
    for  key in dictArg:
        print "dictArg %s-->%s" %(key,dictArg[key])

myList=["my1","my2"]
myDict={"name":"Tom","age":22}
print "1"
foo("formal_args",arg2="argSecond",a=1)
print "*"*40
print "2"
foo(123,"no",myList,myDict)
print "*"*40
print "3"
foo(123,rt=123,*myList,**myDict)
x=""
for i in xrange(45):
    x+="f%d,\'|\',"%i
print x
print "-"*40

a=18
a_list=[1,5,9,10]
val_str="a"
print get_index_by_value(eval(val_str), eval(val_str+"_list"))


remark0="abcde米源fsdfs"
remark = remark0.decode("utf8")
filt = re.compile(u"[\u4e00-\u9fff]+")
filtered_remark = filt.sub(r'', remark)
print filtered_remark

t1=time.strptime("20160818181317","%Y%m%d%H%M%S")
print t1
t1.tm_hour

tt=datetime.datetime(t1.tm_year,t1.tm_mon,t1.tm_mday,2)
#print time.strftime("%Y%m%d%H%M%S",tt)
print tt.strftime("%Y%m%d%H%M%S")

print "------------------------"
type="clk"
day=3
#eval(type+"_"+str(day)+"d=3;")
#print "clk_3d:"+eval(type+"_"+str(day)
locals()['clk_%dd' % day]= day

yy=locals();
print "locals:"
#print "clk_3d:"+str(eval("clk_3d"))
print "clk_3d:"+str(clk_3d)

print "--------------------"
t1=time.strptime("20161031","%Y%m%d")
t2=time.strptime("20161106","%Y%m%d")
print t1
print t2

#
start_date = datetime.datetime.strptime("20161031","%Y%m%d")  
end_date = datetime.datetime.strptime("20161106","%Y%m%d")  
print (end_date - start_date).days


