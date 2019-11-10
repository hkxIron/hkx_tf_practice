import pandas as pd
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.cross_validation import train_test_split

canceData = load_breast_cancer()
X = canceData.data
y = canceData.target
print("X.shape:", X.shape, " y:", y.shape, " type x:",type(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'learning_rate': 0.1,
    'num_leaves': 30,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

data_train = lgb.Dataset(X_train, y_train)
cv_results = lgb.cv(params, data_train,
                    num_boost_round=1000, nfold=5,
                    stratified=True,
                    shuffle=True,
                    metrics='auc',
                    early_stopping_rounds=50,
                    seed=0)
print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())
