import os

from matplotlib import pylab
import numpy as np
import math
from numpy import linalg


from sklearn import linear_model, decomposition

logistic = linear_model.LogisticRegression()


CHART_DIR = os.path.join(".", "charts")

np.random.seed(3)

x1 = np.arange(0, 10, .2)
# x2在x1的基础上进行扰动
x2 = x1 + np.random.normal(loc=0, scale=1, size=len(x1))

def my_pca(X, k_components=2, whiten=False):
    """
    X = U*S*(V^T)
    sigma = 1/N*(X^T)*X =(1/N)* V*S*(U^T)*U*S*V^T
          = (1/N)* V*(S*S)*(V^T) = V*(S*S/N)*(V^T)
    我们只要得到变换矩阵V就行
    那么新的数据:
    X_new = X * V = U * S * V^T * V = U * S
    V:[D,D]
    X_new=X*V[:,K]: [N,K]

    pca需要分析X的协方差的主成份
    :param X: N*D
    :param k_components:要保留的主成份的个数
    :return:
    """
    n_samples = X.shape[0]
    X_normal = X - np.mean(X, axis=0)
    # 注意:SVD中,U, V并非相同维数
    # X_normal: N*D, U:N*D, S:D*D, VT:D*D
    U,S,VT = linalg.svd(X_normal)
    if whiten:
        """
         下面的变换,十分巧妙
         X_new = X*V/sqrt(S*S/n_samples) 
         = X * V/S * sqrt(n_samples))
         = (U*S*V^T)*V/S*sqrt(n_samples)
         = U*(S/S)*V^T*V*sqrt(n_samples)
         = U * sqrt(n_samples)
        """
        X_new = U[:,:k_components]*math.sqrt(n_samples-1)
    else:
        # X_new = X * V = U * S * V^T * V = U * S
        X_new = U[:,:k_components]*S[:k_components]
        # 或者
        #X_new = X_normal @ VT.T[:, 0:k_components] # 取前k个元素

    print("VT shape:", VT.shape, " S:",S.shape, " X_new:", X_new.shape)
    return X_new, S[:k_components]**2/(n_samples-1)

def plot_simple_demo_my_pca():
    pylab.clf()
    fig = pylab.figure(num=None, figsize=(20, 4))
    pylab.subplot(141)

    title = "Original feature space"
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    np.random.seed(3)
    x1 = np.arange(0, 10, .2)
    x2 = x1 + np.random.normal(loc=0, scale=1, size=len(x1))

    # 两者中,有一个>5就是good
    good = (x1 > 5) | (x2 > 5)
    bad = ~good

    x1g = x1[good]
    x2g = x2[good]
    pylab.scatter(x1g, x2g, edgecolor="blue", facecolor="blue")

    x1b = x1[bad]
    x2b = x2[bad]
    pylab.scatter(x1b, x2b, edgecolor="red", facecolor="white")

    pylab.grid(True)

    pylab.subplot(142)

    X = np.c_[(x1, x2)] # X:[N, 2]

    Xtrans, eigen_values = my_pca(X, k_components=2,whiten=False)
    Xtrans_whiten, _ = my_pca(np.copy(X), k_components=2, whiten=True)

    Xg = Xtrans[good]
    Xb = Xtrans[bad]

    print("======手工pca==========")
    print("原始特征值 eigen_values:", eigen_values)
    print("手工Xg:", Xg)
    print("手工Xg(whiten):", Xtrans_whiten[good])

    pylab.scatter(Xg[:, 0], Xg[:,1], edgecolor="blue", facecolor="blue")
    pylab.scatter(Xb[:, 0], Xb[:,1], edgecolor="red", facecolor="white")
        #Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")
    title = "Transformed feature space"
    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[1].get_yaxis().set_visible(True)
    fig.axes[1].set_xlim(-6,6)
    fig.axes[1].set_ylim(-6,6)
    #fig.axes.set_ylim(-6,6)

    # 白化
    Xg = Xtrans_whiten[good]
    Xb = Xtrans_whiten[bad]
    title = "Transformed feature space(whiten)"
    pylab.subplot(143)
    pylab.scatter( Xg[:, 0], Xg[:, 1], edgecolor="blue", facecolor="blue")
    pylab.scatter( Xb[:, 0], Xb[:, 1], edgecolor="red", facecolor="white")

    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[2].get_yaxis().set_visible(True)
    fig.axes[2].set_xlim(-6,6)
    fig.axes[2].set_ylim(-6,6)

    title = "Transformed feature space(1-d)"
    pylab.subplot(144)
    pylab.scatter(Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter(Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")

    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[3].get_yaxis().set_visible(False)
    fig.axes[3].set_xlim(-6,6)
    fig.axes[3].set_ylim(-6,6)


    pylab.grid(True)
    pylab.autoscale(tight=True)
    filename = "my_pca_demo.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")
    # 从图中可以看出我们的pca与 decomposition.PCA的结果完全 一样

def plot_simple_demo_1():
    pylab.clf()
    fig = pylab.figure(num=None, figsize=(10, 4))
    pylab.subplot(131)

    title = "Original feature space"
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    np.random.seed(3)
    x1 = np.arange(0, 10, .2)
    x2 = x1 + np.random.normal(loc=0, scale=1, size=len(x1))

    good = (x1 > 5) | (x2 > 5)
    bad = ~good

    x1g = x1[good]
    x2g = x2[good]
    pylab.scatter(x1g, x2g, edgecolor="blue", facecolor="blue")

    x1b = x1[bad]
    x2b = x2[bad]
    pylab.scatter(x1b, x2b, edgecolor="red", facecolor="white")

    pylab.grid(True)

    pylab.subplot(132)

    X = np.c_[(x1, x2)] # X:[N, 2]

    pca = decomposition.PCA(n_components=2)
    Xtrans = pca.fit_transform(X) # Xtrans:[N, 1], 只有一个主成份
    #Xtrans *=-1

    Xg = Xtrans[good]
    Xb = Xtrans[bad]
    """
    pylab.scatter( Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter( Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")
    """
    pylab.scatter( Xg[:, 0], Xg[:,1], edgecolor="blue", facecolor="blue")
    pylab.scatter( Xb[:, 0], Xb[:,1], edgecolor="red", facecolor="white")
    title = "Transformed feature space"
    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[1].get_yaxis().set_visible(False)

    title = "Transformed feature space(1-d)"
    pylab.subplot(133)
    pylab.scatter( Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter( Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")

    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[2].get_yaxis().set_visible(False)

    print("======系统pca 数据1==========")
    print("特征值:",pca.explained_variance_)
    print("特征重要度比率:",pca.explained_variance_ratio_)
    # 可以看出, 手工pca与系统pca数据有个符号相反,并不影响分解,
    # 比如 X=U*S*(V^T) = (-U) S*(-V^T)
    # 对于二维数据, 其物理意义对应是按照顺时针旋转还是逆时针旋转,并不影响保持最大分量这一物理意义
    print("系统Xg:", Xg)
    pylab.grid(True)

    pylab.autoscale(tight=True)
    filename = "pca_demo_1.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


def plot_simple_demo_2():
    pylab.clf()
    fig = pylab.figure(num=None, figsize=(10, 4))
    pylab.subplot(121)

    title = "Original feature space"
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    np.random.seed(3)
    x1 = np.arange(0, 10, .2)
    x2 = x1 + np.random.normal(loc=0, scale=1, size=len(x1))

    good = x1 > x2
    bad = ~good

    x1g = x1[good]
    x2g = x2[good]
    pylab.scatter(x1g, x2g, edgecolor="blue", facecolor="blue")

    x1b = x1[bad]
    x2b = x2[bad]
    pylab.scatter(x1b, x2b, edgecolor="red", facecolor="white")

    pylab.grid(True)

    pylab.subplot(122)

    X = np.c_[(x1, x2)]

    pca = decomposition.PCA(n_components=1)
    Xtrans = pca.fit_transform(X)

    Xg = Xtrans[good]
    Xb = Xtrans[bad]

    pylab.scatter(
        Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter(
        Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")
    title = "Transformed feature space"
    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[1].get_yaxis().set_visible(False)

    print("======系统pca 数据2==========")
    print("特征重要度:",pca.explained_variance_ratio_)

    pylab.grid(True)

    pylab.autoscale(tight=True)
    filename = "pca_demo_2.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")


def plot_simple_demo_lda():
    #from sklearn import lda
    # 与demo2不同的是,lda会利用(即有监督的标签信息,而pca并不会利用此信息)类别信息进行降维
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pylab.clf()
    fig = pylab.figure(num=None, figsize=(10, 4))
    pylab.subplot(121)

    title = "Original feature space"
    pylab.title(title)
    pylab.xlabel("$X_1$")
    pylab.ylabel("$X_2$")

    good = x1 > x2
    bad = ~good

    x1g = x1[good]
    x2g = x2[good]
    pylab.scatter(x1g, x2g, edgecolor="blue", facecolor="blue")

    x1b = x1[bad]
    x2b = x2[bad]
    pylab.scatter(x1b, x2b, edgecolor="red", facecolor="white")

    pylab.grid(True)

    pylab.subplot(122)

    X = np.c_[(x1, x2)]

    #lda_inst = lda.LDA(n_components=1)
    lda_inst = LinearDiscriminantAnalysis(n_components=1)
    Xtrans = lda_inst.fit_transform(X, good)

    Xg = Xtrans[good]
    Xb = Xtrans[bad]

    pylab.scatter(
        Xg[:, 0], np.zeros(len(Xg)), edgecolor="blue", facecolor="blue")
    pylab.scatter(
        Xb[:, 0], np.zeros(len(Xb)), edgecolor="red", facecolor="white")
    title = "Transformed feature space"
    pylab.title(title)
    pylab.xlabel("$X'$")
    fig.axes[1].get_yaxis().set_visible(False)

    pylab.grid(True)

    pylab.autoscale(tight=True)
    filename = "lda_demo.png"
    pylab.savefig(os.path.join(CHART_DIR, filename), bbox_inches="tight")

if __name__ == '__main__':
    plot_simple_demo_my_pca()
    plot_simple_demo_1()
    plot_simple_demo_2()
    plot_simple_demo_lda()
