# -*- coding: utf-8 -*-  
# This script generates web traffic data for our hypothetical
# web startup "MLASS" in chapter 01

import os
import scipy as sp
from scipy.stats import gamma
import matplotlib.pyplot as plt

sp.random.seed(3)  # to reproduce the data later on

x = sp.arange(1, 31 * 24) #743*1
y = sp.array(200 * (sp.sin(2 * sp.pi * x / (7 * 24))), dtype=int) #743*1
y += gamma.rvs(15, loc=0, scale=100, size=len(x))  #Random variates of given type.
y += 2 * sp.exp(x / 100.0)
y = sp.ma.array(y, mask=[y < 0])  #Masked values of True exclude the corresponding element from any computation,将小于0的元素剔除掉，不参与计算
print(sum(y), sum(y < 0))

plt.scatter(x, y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w * 7 * 24 for w in [0, 1, 2, 3, 4]], ['week %i' % (w + 1) for w in [
           0, 1, 2, 3, 4]])

plt.autoscale(tight=True)
plt.grid()
plt.savefig(os.path.join("..", "1400_01_01.png")) #保存为图片

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data")

# sp.savetxt(os.path.join("..", "web_traffic.tsv"),
# zip(x[~y.mask],y[~y.mask]), delimiter="\t", fmt="%i")
sp.savetxt(os.path.join(
    data_dir, "web_traffic.tsv"), list(zip(x, y)), delimiter="\t", fmt="%s")