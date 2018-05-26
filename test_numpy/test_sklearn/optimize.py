#coding:utf-8

from scipy.optimize import fmin,fmin_bfgs
#最优化，求解方程的最优解,即最小值

def myfunc(x):
	return x**2-4*x+8 #

x0 = [1.3] #猜一个初值
xopt = fmin(myfunc, x0) #求解
print xopt #打印结果运行之后，


xopt3 = fmin_bfgs(myfunc, x0)
print xopt3