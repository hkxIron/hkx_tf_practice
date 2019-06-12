import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
#%matplotlib inline

"""
首先，我们生成一组随机数据，为了体现DBSCAN在非凸数据的聚类优点，我们生成了三簇数据，两组是非凸的。代码如下：
"""
X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                      noise=.05)
X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
               random_state=9)

X = np.concatenate((X1, X2))
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

# 首先我们看看K-Means的聚类效果，代码如下：
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("k-means")
plt.show()

# 那么如果使用DBSCAN效果如何呢？我们先不调参，直接用默认参数，看看聚类效果,代码如下
from sklearn.cluster import DBSCAN
y_pred = DBSCAN().fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("dbscan(default hp)")
plt.show()


# 怎么办？看来我们需要对DBSCAN的两个关键的参数eps和min_samples进行调参！从上图我们可以发现，类别数太少，我们需要增加类别数，那么我们可以减少ϵ-邻域的大小，默认是0.5，我们减到0.1看看效果。代码如下：

y_pred = DBSCAN(eps = 0.1).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("dbscan(eps=0.1)")
plt.show()

#　可以看到聚类效果有了改进，至少边上的那个簇已经被发现出来了。此时我们需要继续调参增加类别，有两个方向都是可以的，一个是继续减少eps，另一个是增加min_samples。我们现在将min_samples从默认的5增加到10，代码如下：

y_pred = DBSCAN(eps = 0.1, min_samples = 10).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("dbscan(eps=0.1, min_samples=10)")
plt.show()