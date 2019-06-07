# -*-encoding:utf-8 -*-
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D as ax3
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, fetch_olivetti_faces, load_digits

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler
from sklearn import decomposition
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

color = ['red', 'purple', 'yellow']
for i in colors.cnames:
    if i in color:
        pass
    else:
        color.append(i)


def generate_circle_data3():
    xx = np.zeros((2700, 3))
    x1 = np.ones((900,)) + 0.5 * np.random.rand(900) - 0.5
    r1 = np.linspace(0, 2 * np.pi, 30)
    r2 = np.linspace(0, np.pi, 30)
    r1, r2 = np.meshgrid(r1, r2)
    r1 = r1.ravel()
    r2 = r2.ravel()
    xx[0:900, 0] = x1 * np.sin(r1) * np.sin(r2)
    xx[0:900, 1] = x1 * np.cos(r1) * np.sin(r2)
    xx[0:900, 2] = x1 * np.cos(r2)
    x1 = 3 * np.ones((900,)) + 0.6 * np.random.rand(900) - 0.6
    xx[900:1800, 0] = x1 * np.sin(r1) * np.sin(r2)
    xx[900:1800, 1] = x1 * np.cos(r1) * np.sin(r2)
    xx[900:1800, 2] = x1 * np.cos(r2)
    x1 = 6 * np.ones((900,)) + 1.1 * np.random.rand(900) - 0.6
    xx[1800:2700, 0] = x1 * np.sin(r1) * np.sin(r2)
    xx[1800:2700, 1] = x1 * np.cos(r1) * np.sin(r2)
    xx[1800:2700, 2] = x1 * np.cos(r2)
    target = np.zeros((2700,))
    target[0:900] = 0
    target[900:1800] = 1
    target[1800:2700] = 2
    target = target.astype('int')
    return xx, target


def compare_KPCA():
    data, target = generate_circle_data3()
    pca = decomposition.PCA(n_components=2)
    data1 = pca.fit_transform(data)
    try:
        figure1 = plt.figure(1)
        ax = ax3(figure1)
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=[color[i] for i in target], alpha=0.5)
        plt.title('Origin Data')
    except:
        pass

    figure2 = plt.figure(2)
    k = 1
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']: # 线性核即是原始的pca
        plt.subplot(1, 4, k)
        k += 1
        kpca = decomposition.KernelPCA(n_components=2, kernel=kernel)
        data_reduced = kpca.fit_transform(data)
        plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=[color[i] for i in target])
        plt.title(kernel)
    plt.suptitle('The Comparasion Between KPCA')
    plt.show()


def bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


compare_KPCA()