import xgboost as xgb
# doc: https://xgboost.readthedocs.io/en/latest/python/python_intro.html#data-interface
#dtrain = xgb.DMatrix('train.svm.txt')
dtrain = xgb.DMatrix('train2.svm.txt')

#param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic', "booster":"gbtree"}
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}


num_round = 10
bst = xgb.train(param, dtrain, num_round)
