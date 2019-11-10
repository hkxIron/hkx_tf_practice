#%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression


def hidden_model(x):
    # y is a linear combination of columns 5 and 10...
    result = x[:, 5] + x[:, 10]
    # ... with a little noise
    result += np.random.normal(0, .005, result.shape)
    return result


def make_x(nobs):
    return np.random.uniform(0, 3, (nobs, 10 ** 6))

x = make_x(100)
y = hidden_model(x)  # 用了col5与col10的特征作为目标

# k:Number of top features to select.
# f_regression:
# Linear model for testing the individual effect of each of many regressors.
# This is a scoring function to be used in a feature seletion procedure, not
# a free standing feature selection procedure.
# This is done in 2 steps:
#
# 1. The correlation between each regressor and the target is computed, that is, ((X[:, i] - mean(X[:, i])) * (y - mean_y)) / (std(X[:, i]) * std(y)).
# 2. It is converted to an F score then to a p-value.

selector = SelectKBest(f_regression, k=2).fit(x, y)
best_features = np.where(selector.get_support())[0]
print(best_features)

x2 = x[:, best_features]
clf = LinearRegression().fit(x2, y)
y2p = clf.predict(x2)

scores = []
#for train, test in KFold(len(y), n_splits=5):
for train, test in KFold(n_splits=5).split(x):
    xtrain, xtest, ytrain, ytest = x[train], x[test], y[train], y[test]

    # 1.进行特征选择
    b = SelectKBest(f_regression, k=2) #选择最好的两个特征列
    b.fit(xtrain, ytrain)
    print(b.get_support()) # [False False False ... False False False]
    
    # 2.使用选择后的特征进行训练
    xtrain = xtrain[:, b.get_support()]
    xtest = xtest[:, b.get_support()]
    clf.fit(xtrain, ytrain)
    scores.append(clf.score(xtest, ytest))

    yp = clf.predict(xtest)
    
    plt.plot(yp, ytest, 'o') # test集上预测
    
    plt.plot(ytest, ytest, 'r-') # 真实的label

plt.xlabel("Predicted")
plt.ylabel("Observed")

print("CV Score is ", np.mean(scores))
