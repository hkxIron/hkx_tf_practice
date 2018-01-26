from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f1():
    print(stats.norm.stats())
    print("fast")
    x = stats.norm.rvs(size=10000)
    print(np.mean(x),np.var(x))

    t= np.arange(-10,10,0.01)

    fig1 = plt.figure(2)
    rects =plt.bar(left = (0.2,1),height = (1,0.5),width = 0.2,align="center",yerr=0.000001)
    plt.title('Pe')
    plt.show()

def f2():
    #概率分布直方图
    #高斯分布
    #均值为0
    mean = 0
    #标准差为1，反应数据集中还是分散的值
    sigma = 1
    x=mean+sigma*np.random.randn(10000)
    fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
    #第二个参数是柱子宽一些还是窄一些，越大越窄越密
    ax0.hist(x,40,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
    ##pdf概率分布图，一万个数落在某个区间内的数有多少个
    ax0.set_title('pdf')
    ax1.hist(x,20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)
    #cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
    ax1.set_title("cdf")
    fig.subplots_adjust(hspace=0.4)
    plt.show()

def f3():
    mean = 0
    #标准差为1，反应数据集中还是分散的值
    sigma = 1
    x=mean+sigma*np.random.randn(10000)
    #第二个参数是柱子宽一些还是窄一些，越大越窄越密
    plt.hist(x,bins=40,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
    plt.title("hist")
    ##pdf概率分布图，一万个数落在某个区间内的数有多少个
    plt.show()

def f4():
    import scipy as sp
    sp.stats.binom.pmf(k=range(6),args=5,kwrds=1/6.0)
    tt="sdf|123"
    tt.rstrip("\n")


if __name__ =='__main__' :
    f3()