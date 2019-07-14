from mlxtend.regressor import StackingCVRegressor
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
RANDOM_SEED = 42
print("loading data...")
X, y = load_boston(return_X_y=True)

print("create svr ...")
svr = SVR(kernel='linear')
print("create lasso ...")
lasso = Lasso()
rf = RandomForestRegressor(n_estimators=5,
random_state=RANDOM_SEED)
# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)
stack = StackingCVRegressor(regressors=(svr, lasso, rf),
meta_regressor=lasso)# 注意,最后融合的是lasso算法

print('5-fold cross validation scores:\n')
for clf, name in zip([svr, lasso, rf, stack], ['SVM', 'Lasso', 'Random Forest', 'StackingClassifier']):
	scores = cross_val_score(clf, X, y, cv=5)
	print("R^2 Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), name))

'''
输出:
5-fold cross validation scores:
R^2 Score: 0.45 (+/- 0.29) [SVM]
R^2 Score: 0.43 (+/- 0.14) [Lasso]
R^2 Score: 0.52 (+/- 0.28) [Random Forest]
R^2 Score: 0.58 (+/- 0.24) [StackingClassifier]
'''


