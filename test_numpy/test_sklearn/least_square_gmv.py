#coding:utf-8
from pylab import *
from numpy import *
from scipy.optimize import leastsq
import sys
#详见  E:\kp\docs\机器学习话题\最优化\python 最优化
#fp是真实的函数，v是系数，x是变量。fn是fp函数的一个具体实现，e是误差函数。
#fp = lambda v, x: v[0]*x**2+v[1]*x+v[2]
fp = lambda v, x: v[0]*x**3+v[1]*x**2+v[2]*x+v[3]
e = lambda v, x, y: (fp(v,x)-y)**2  #**2 #误差,y为目标
#设计一个含有噪音的函数
n=30
xmin=0
xmax=10
x=array([0,1,2,3,4,5]) #从2010年开始
y=array([9,52,191,350,571,912]) #gmv目标
#y = fn(x) + rand(len(x))*0.2*(fn(x).max()-fn(x).min())
#设计初始值，并调用leastsq函数求解。用args指明变量，maxfev是最多调用函数的次数。
#v0 = [3., 1, 4] #为v的初始值
v0 = [3., 1, 4,1] #为v的初始值
v, success = leastsq(e, v0, args=(x,y), maxfev=10000) #v为求得的最佳参数
print "estimate parameters:"
print v

print "rmse:"
print sqrt(sum(e(v,x,y)))/len(x)


#绘出真实曲线和拟合曲线
x_extend=arange(10)
for year in x_extend:
    sys.stdout.write("201%d年gmv:%f "%(year,fp(v,year)))
    if year<len(y):
        sys.stdout.write(" real:%f"%y[year])
    print ""

def plot_fit():
    print 'Estimater parameters: ', v
    X = linspace(xmin,xmax,n*5) #X为拟合的曲线准备的，所以点的个数要多一些
    plot(x,y,'ro', # 原始点
         X, fp(v,X))  #拟合出的点
    
    
plot_fit()
show()

