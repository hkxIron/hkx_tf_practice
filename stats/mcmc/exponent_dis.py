# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def getExponential(SampleSize,p_lambda):
    result = -np.log(1-np.random.uniform(0,1,SampleSize))/p_lambda
    return result

# 生成10000个数，观察它们的分布情况
SampleSize = 10000
es = getExponential(SampleSize, 1)
plt.hist(es,np.linspace(0,5,50),facecolor="green")
plt.show()