# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
# https://bindog.github.io/blog/2015/05/20/different-method-to-generate-normal-distribution/
def getNormal(SampleSize,n):
    xsum = []
    for i in range(SampleSize):
        # 利用中心极限定理，[0,1)均匀分布期望为0.5，方差为1/12
        """
        根据中心极限定理，生成正态分布就非常简单粗暴了，直接生成n个独立同分布的随机变量，求和即可。
        注意，无论你使用什么分布，当n趋近于无穷大时，它们和的分布都会趋近正态分布！
        """
        # tsum ~ N(0,1)
        tsum = (np.mean(np.random.uniform(0,1,n))-0.5)*np.sqrt(12*n)
        xsum.append(tsum)
    return xsum

# 生成10000个数，观察它们的分布情况
SampleSize = 10000
# 观察n选不同值时，对最终结果的影响
N = [1,2,10,1000]

plt.figure(figsize=(10,20))
subi = 220
for index,n in enumerate(N):
    subi += 1
    plt.subplot(subi)
    normalsum = getNormal(SampleSize, n)
    # 绘制直方图
    plt.hist(normalsum,np.linspace(-4,4,80),facecolor="green",label="n={0}".format(n))
    plt.ylim([0,450])
    plt.legend()

plt.show()
