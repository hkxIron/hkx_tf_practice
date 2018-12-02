# blog:http://nbviewer.jupyter.org/github/Edward1Chou/textClustering/blob/master/experiment/DBSCANtest.ipynb
# https://github.com/Edward1Chou/textClustering/tree/master/experiment

from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
# Generate isotropic Gaussian blobs for clustering
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)
print("X:", X)
print("labels_true:", labels_true)

import numpy as np
from sklearn.cluster import DBSCAN


# Compute DBSCAN
db = DBSCAN(eps=0.2, min_samples=10).fit(X)
print("db labels:", db.labels_)
print("core_sample_indices_:", db.core_sample_indices_)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool) # 是否是核心点
print("core_samples_mask:", core_samples_mask)

core_samples_mask[db.core_sample_indices_] = True
print("core_samples_mask:", core_samples_mask)

labels = db.labels_

from sklearn import metrics
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))
# 画图
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, color in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        color = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()




