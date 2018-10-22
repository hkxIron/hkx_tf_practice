# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as ss

# blog: http://bindog.github.io/blog/2015/10/15/revisit-mcmc-method/
"""
输入:已知待采样分布的概率密度
输出:采样出符合此概率密度的一组样本 

0x02 MCMC模拟Beta分布

在花式生成正态分布中，提到了一种拒绝采样的方法，这背后蕴含的其实就是蒙特卡洛方法的思想。
这里以Beta分布为例，Beta分布的概率密度函数PDF是f(x) = C*[x^(a-1)]*[(1-x)^(b-1)] ，其中C=1/B(a,b) ，
回顾《花式生成正态分布》里面提到的方法，如果采用反变换法，则需要对这个函数求积分和反函数，非常麻烦。
如果采用拒绝采样就简单直观了，对照千岛湖植物考察的例子，为了对Beta分布进行采样，需要定义转移概率矩阵和接受概率，
这里可以忽略转移概率矩阵(不同状态之间的转移概率是相同的, pji==pij)，只考虑接受概率
aij = min(1,pj*pji/(pi*pij)) = min(1, pj/pi)

pi和pj为平稳分布概率，与Beta分布的概率密度是一致的，因此
pi = C*i^(a-1)*(1-i)^(b-1)
pj = C*j^(a-1)*(1-j)^(b-1)

"""
def test_mcmc():

    # 真实Beta分布概率密度函数
    def beta(x, a, b):
        return (1.0 / ss.beta(a,b)) * x**(a-1) * (1-x)**(b-1)

    # Beta分布概率密度函数(忽略了Beta函数前面的系数,因为其与x无关)
    # 在mcmc采样中,我们已知 待采样分布的概率密度
    def beta_fpdf(x,a,b):
        return x**(a-1) * (1-x)**(b-1)

    # 根据接受概率决定是否转移
    def transform(ap):
        stone = random.uniform(0,1)
        # 如果在转移概率之内，就进行转移，否则不转移
        if stone>=ap:
            return False
        else:
            return True

    def beta_mcmc(N_hops,a,b):
        states = []
        cur = random.uniform(0,1)
        for i in range(0,N_hops):
            states.append(cur)
            next = random.uniform(0,1)
            # a(i->j) = p(j)p(j->i)/[p(i)p(i->j)], 假定p(j->i)与p(i->j)相等
            accept_prob = min(beta_fpdf(next, a, b) / beta_fpdf(cur, a, b), 1) # 计算接受概率
            if transform(accept_prob):
                cur = next
        return states[-5000:] # 返回进入平稳分布后的1000个状态,收敛之前的状态称为 burn-in period


    # 绘制通过MCMC方法抽样的Beta分布
    def plot_beta(a, b):
        Ly = []
        Lx = []
        i_list = np.mgrid[0:1:100j] # 当第3个参数为虚数时，表示返回数组的长度
        # print("i_list:",i_list)
        for i in i_list:
            Lx.append(i)
            Ly.append(beta(i, a, b)) # 真实的beta分布
        # 绘制真实的Beta分布进行对照
        plt.plot(Lx, Ly, label="Real Distribution: a="+str(a)+", b="+str(b))
        # mcmc 采样得到的分布
        mcmc_data = beta_mcmc(200000, a,b)
        plt.hist(mcmc_data, normed=True,bins=100, histtype='step',label="Simulated_MCMC: a="+str(a)+", b="+str(b))
        plt.legend()
        plt.show()

    plot_beta(0.5, 0.6)


#通过采样的方法计算积分
def test_integate():
    import random

    def samplexyz():
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        z = random.uniform(0, 1)
        return x, y, z

    def fx(x, y, z):
        return x

    multidimi = 0.0
    n = 100000
    for i in range(n):
        x, y, z = samplexyz()
        if x + y + z <= 1:
            ti = fx(x, y, z)
            multidimi += ti

    simulate_value = multidimi/n
    real_value =1.0/24
    print("simulate_value:%f real_value:%f"%(simulate_value,real_value))

#test_integate()
test_mcmc()