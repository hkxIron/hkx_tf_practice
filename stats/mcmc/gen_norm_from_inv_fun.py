# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def getNormal(SampleSize):
    iid = np.random.uniform(0,1,SampleSize)
    # Percent point function (inverse of `cdf`) at q of the given RV
    # 通过正态分布的cdf的逆函数生成其分布本身
    result = norm.ppf(iid)
    return result

SampleSize = 10000000
normal = getNormal(SampleSize)
plt.hist(normal,np.linspace(-4,4,81),facecolor="green")
plt.show()