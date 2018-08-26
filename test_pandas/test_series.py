# 详细tutorial请见: https://github.com/hkxIron/pydata-notebook
# 英文版 https://github.com/wesm/pydata-book

from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
import os
np.random.seed(12345)
obj = Series([4, 7, -5, 3])
print("obj:", obj)
print("obj.values:", obj.values)
print("obj.index:", obj.index)

#-------------
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
partialRows = obj2[['c', 'a', 'd']] # 选取Series中的部分行
print("partial rows:\n", partialRows)

