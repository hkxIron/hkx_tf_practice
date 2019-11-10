print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

def f1():
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    """
    类别名:
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
    print("class y:", np.unique(y))
    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    n_samples, n_features = X.shape # 150, 3
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)] #按列stack

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other, One-Vs-all只需要训练K个分类器,而不是K*(K-1)个
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear',
                                             probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes): #计算每类上的tpr,fpr
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    print("y_test shape:", y_test.shape) # shape:[75, 3]
    # ravel后,一个样本就会有k次分类了, k=类别数
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true=y_test.ravel(), y_score=y_score.ravel()) # 计算所有样本对所有类的auc
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # plot auc
    plt.figure()
    lw = 2 # 类别virginica
    plt.plot(fpr[2], tpr[2],
             color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example(for classs=virginica)')
    plt.legend(loc="lower right")
    plt.show()

    print("画出多类别的auc图")
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)], axis=0))
    print("all_fpr:", all_fpr)

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(x=all_fpr, xp=fpr[i], fp=tpr[i]) #对每类tpr进行插值

    # Finally average it and compute AUC
    mean_tpr /= n_classes # 求平均

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # 画出每类的auc曲线
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

f1()

