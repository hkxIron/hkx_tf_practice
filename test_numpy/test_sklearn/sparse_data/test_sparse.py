
"""
大文件:
for chunk in pd.read_csv('./data/small_training.txt',sep='\t',header=None,index_col=False, chunksize=11):
  proccess(chunk)

"""

import numpy as np
import pandas as pd

np.random.seed(seed=12)  ## for reproducibility
dataset = np.random.binomial(1, 0.01, 20000000).reshape(2000, 10000)  ## dummy data
y = np.random.binomial(1, 0.5, 2000)  ## dummy target variable

import matplotlib.pyplot as plt
plt.spy(dataset)
plt.title("Sparse Matrix")
#plt.show()

from scipy.sparse import csr_matrix
sparse_dataset = csr_matrix(dataset)

from sklearn.naive_bayes import BernoulliNB

nb = BernoulliNB(binarize=None)
import time
t1 = time.time()
nb.fit(dataset, y)
t2 = time.time()

nb.fit(sparse_dataset, y)
t3 = time.time()
print("dense data:", t2 - t1)
print("sparse data:", t3 - t2)
"""
时间相差很多倍
dense data: 0.13862872123718262
sparse data: 0.0049855709075927734
"""
