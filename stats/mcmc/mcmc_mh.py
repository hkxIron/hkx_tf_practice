# blog: https://www.cnblogs.com/pinard/p/6638955.html
import random
import math
import numpy as np
from scipy.stats import norm
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.special as ss

"""
在Gibbs采样过程中,最终得到的样本就是,马氏链收敛过程中,最终得到的样本为p(x,y),而收敛之前的阶段为burn-in period 

M-H采样是Metropolis-Hastings采样的简称，这个算法首先由Metropolis提出，被Hastings改进，因此被称之为Metropolis-Hastings采样或M-H采样
M-H采样解决了我们上一节MCMC采样接受率过低的问题。

本例中，要模拟的是正态分布的样本
在例子里，我们的目标平稳分布是一个均值3，标准差2的正态分布，

而选择的马尔可夫链状态转移矩阵Q(i,j)的条件转移概率是以i为均值,方差1的正态分布在位置j的值。

这个例子仅仅用来让大家加深对M-H采样过程的理解。
毕竟一个普通的一维正态分布用不着去用M-H采样来获得样本。

"""
def metropolis_hastings_beta_sampling():
    # 模拟beta分布
    a=0.5
    b=0.6
    def beta_dist_prob(x):
        return (1.0 / ss.beta(a, b)) * x ** (a - 1) * (1 - x) ** (b - 1)

    T = 10000
    #pi = np.random.random_sample(T)
    pi = [random.random() for _ in range(T)]
    sigma = 1
    t = 0
    while t < T-1:
        t = t + 1
        # 1. 从条件概率分布Q(x|xt)中采样得到样本x∗
        # pi_star为只有一个元素的列表,从正态分布中选出一个元素X
        # 转移矩阵Q(i,j)的条件转移概率是以i为均值,方差1的正态分布在位置j的值 (i->j)
        pi_star = random.uniform(0,1)
        # 2. 从均匀分布采样u∼uniform[0,1]
        u = random.uniform(0, 1)
        # pi[t-1]为上一时刻的采样元素
        # p(x)p(y|x) /(p(y)P(x|y)) ,貌似此处认为转移概率p(y|x)与p(x|y)相等
        alpha = min(1, (beta_dist_prob(pi_star) / beta_dist_prob(pi[t - 1]))) # 接受率

        #3.如果u<α(xt,x∗)=min{π(j)Q(j,i)/(π(i)Q(i,j)),1}, 则接受转移xt→x∗，即xt+1=x∗
        #  否则不接受转移，t=max(t−1,0)
        if u < alpha: # 接受新元素
            pi[t] = pi_star
        else: # 拒绝新元素,维持老元素
            pi[t] = pi[t - 1]
    # 画出采样后元素的分布直方图
    real_y = [ beta_dist_prob(x) for x in pi ]
    plt.scatter(pi, real_y, edgecolors='black')

    num_bins = 50
    plt.hist(pi, num_bins, normed=1, facecolor='red', alpha=0.7)
    plt.show()

def metropolis_hastings_sampling():
    # 计算theta处的概率密度
    def norm_dist_prob(theta):
        y = norm.pdf(theta, loc=3, scale=2)  # 目标分布的均值为3,标准差为2
        return y

    T = 10000
    pi = [0 for _ in range(T)]
    sigma = 1
    t = 0
    while t < T-1:
        t = t + 1
        # 1. 从条件概率分布Q(x|xt)中采样得到样本x∗
        # pi_star为只有一个元素的列表,从正态分布中选出一个元素X
        # 转移矩阵Q(i,j)的条件转移概率是以i为均值,方差1的正态分布在位置j的值 (i->j)
        # 此处有些问题,本身就是要模拟正态分布,结果产生样本时就已经使用了!! 相当于用结论去证明!所以此种做法是错误的
        #pi_star = norm.rvs(loc= pi[t - 1], scale=sigma, size=1, random_state=None)[0] # 从条件分布Q(x|x_t)中采样,可以看出均值有变化, pi_star只有一个元素
        # 应该利用与目标分布不一样的分布
        #pi_star = norm.rvs(loc=0, scale=1) # 注意,此处的分布产生的样本应尽可能覆盖目标分布的定义域,否则会不准
        # 此处用正态分布以及均匀分布均可以
        pi_star = random.uniform(-10, 10)
        #pi_star = st.gamma.rvs(1, 0.5)
        #pi_star = norm.rvs(loc=0, scale=10) # 注意,此处的分布产生的样本应尽可能覆盖目标分布的定义域,否则会不准
        # 2. 从均匀分布采样u∼uniform[0,1]
        u = random.uniform(0, 1)
        # pi[t-1]为上一时刻的采样元素
        # p(x)p(y|x) /(p(y)P(x|y)) ,貌似此处认为转移概率p(y|x)与p(x|y)相等
        alpha = min(1, (norm_dist_prob(pi_star) / norm_dist_prob(pi[t - 1]))) # 接受率

        #3.如果u<α(xt,x∗)=min{π(j)Q(j,i)/(π(i)Q(i,j)),1}, 则接受转移xt→x∗，即xt+1=x∗
        #  否则不接受转移，t=max(t−1,0)
        if u < alpha: # 接受新元素
            pi[t] = pi_star
        else: # 拒绝新元素,维持老元素
            pi[t] = pi[t - 1]
    # 画出采样后元素的分布直方图
    plt.scatter(pi, norm.pdf(pi, loc=3, scale=2), edgecolors='black')
    #plt.scatter(pi, norm.pdf(pi), edgecolors='black')

    num_bins = 50
    plt.hist(pi, num_bins, normed=1, facecolor='red', alpha=0.7)
    plt.show()



def gibbs_sampling():
    # blog:http://www.cnblogs.com/pinard/p/6645766.html
    # 二维gibbs采样
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import multivariate_normal

    # 多维正态分布
    samplesource = multivariate_normal(mean=[5,-1], cov=[[1,0.5],[0.5,2]])

    # 从分布中采样得到样本y: x-> y
    def p_ygivenx(x, m1, m2, s1, s2):
        return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt(1 - rho ** 2) * s2))

    # 从分布中采样得到样本x: y-> x
    def p_xgiveny(y, m1, m2, s1, s2):
        return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt(1 - rho ** 2) * s1))

    N = 5000
    K = 20
    x_res = []
    y_res = []
    z_res = []
    m1 = 5 # 均值
    s1 = 1 # 标准差

    m2 = -1
    s2 = 2 # 标准差

    rho = 0.5 # rho:线性相关系数
    y = m2

    # gibbs采样:
    for i in range(N):
        for j in range(K):
            x = p_xgiveny(y, m1, m2, s1, s2) # 采样过程中的需要的状态转移条件分布 P(x|y)
            y = p_ygivenx(x, m1, m2, s1, s2) # P(y|x)
            z = samplesource.pdf([x,y]) # 计算其概率密度
            x_res.append(x)
            y_res.append(y)
            z_res.append(z)

    print("x_res size:{}".format(len(x_res))) # 100000
    num_bins = 50
    plt.hist(x_res, num_bins, normed=1, facecolor='green', alpha=0.5)
    plt.hist(y_res, num_bins, normed=1, facecolor='red', alpha=0.5)
    plt.title('Histogram')
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    ax.scatter(x_res, y_res, z_res, marker='+')
    plt.show()

#metropolis_hastings_beta_sampling()
metropolis_hastings_sampling()
#gibbs_sampling()
