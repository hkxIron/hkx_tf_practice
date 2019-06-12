# https://github.com/ljpzzz/machinelearning/blob/master/classic-machine-learning/spectral_cluster.ipynb

import numpy as np
from sklearn import datasets
# 首先我们生成500个个6维的数据集，分为5个簇。由于是6维，这里就不可视化了，代码如下：
X, y = datasets.make_blobs(n_samples=500,
                           n_features=6,
                           centers=5,
                           cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4],
                           random_state=11)

from sklearn.cluster import SpectralClustering
y_pred = SpectralClustering().fit_predict(X)
from sklearn import metrics
# 下面的分数,越大越好
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))

"""
由于我们使用的是高斯核，那么我们一般需要对n_clusters和gamma进行调参。选择合适的参数值。代码如下：
"""
for index, gamma in enumerate((0.01,0.1,1,10)):
    for index, k in enumerate((3,4,5,6)):
        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
        print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:",
              metrics.calinski_harabaz_score(X, y_pred))


"""
　　　　我们可以看看不输入可选的n_clusters的时候，仅仅用最优的gamma为0.1时候的聚类效果，代码如下：
输出为：

Calinski-Harabasz Score 14950.4939717
　　　　可见n_clusters一般还是调参选择比较好。
"""

y_pred = SpectralClustering(gamma=0.1).fit_predict(X)
print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))


