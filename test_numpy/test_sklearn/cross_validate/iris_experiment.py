# blog: https://blog.csdn.net/jasonding1354/article/details/50562513
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()

X = iris.data
y = iris.target

for i in range(1,5):
    print("random_state is ", i,", and accuracy score is:")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))

# k-fold cross validation

# 下面代码演示了K-fold交叉验证是如何进行数据分割的
# simulate splitting a dataset of 25 observations into 5 folds
from sklearn.cross_validation import KFold
kf = KFold(25, n_folds=5, shuffle=False)

# print the contents of each training and testing set
print('{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations'))
for iteration, data in enumerate(kf, start=1):
    #print('{:^9} {} {:^25}'.format(iteration, data[0], data[1]))
    print(iteration, data[0], data[1])

# k-fold cv 调节参数
print("k-fold cv调节参数")
from sklearn.cross_validation import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5)
# 这里的cross_val_score将交叉验证的整个过程连接起来，不用再进行手动的分割数据
# cv参数用于规定将原始数据分成多少份
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)

# use average accuracy as an estimate of out-of-sample accuracy
# 对十次迭代计算平均的测试准确率
print(scores.mean())

# search for an optimal value of K for KNN model
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

print(k_scores)


import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(k_range, k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross validated accuracy")
#plt.show()

print("4.2 用于模型选择")
# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=20)
print(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

# 10-fold cross-validation with logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())

print("4.3 用于特征选择")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
# read in the advertising dataset
#data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
data = pd.read_csv('Advertising.csv', index_col=0)

# create a Python list of three feature names
feature_cols = ['TV', 'radio', 'newspaper']

# use the list to select a subset of the DataFrame (X)
X = data[feature_cols]

# select the Sales column as the response (y)
y = data["sales"]

lm = LinearRegression()
#scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
scores = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')
print(scores)

# fix the sign of MSE scores
mse_scores = -scores
print(mse_scores)

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)

# calculate the average RMSE
print(rmse_scores.mean())

# 10-fold cross-validation with two features (excluding Newspaper)
feature_cols = ['TV', 'radio']
X = data[feature_cols]
print(np.sqrt(-cross_val_score(lm, X, y, cv=10, scoring='neg_mean_squared_error')).mean())



