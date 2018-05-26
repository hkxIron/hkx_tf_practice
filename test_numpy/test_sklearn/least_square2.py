#coding:utf-8
from numpy import *
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


x =arange(0,6e-2,6e-2/30)
A,k,theta = 10, 1.0/3e-2, pi/6
y_true = A*sin(2*pi*k*x+theta) #真实值
y_meas = y_true + 2*random.randn(len(x)) #加入噪声后的值
def residuals(p, y, x):
    A,k,theta = p
    err = y-A*sin(2*pi*k*x+theta)
    return err

def peval_r(x, p):
    return p[0]*sin(2*pi*p[1]*x+p[2])

p0 = [8, 1/2.3e-2, pi/3]  #猜测初始值
plsq = leastsq(residuals, p0, args=(y_meas, x))
plt.plot(x,peval_r(x,plsq[0]), #fit
         x,y_meas, #noisy
         'o',x,y_true) #true
plt.title('Least-squares fit to noisy data')
plt.legend(['Fit', 'Noisy', 'True'])
plt.show()

