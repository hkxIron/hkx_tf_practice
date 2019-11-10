import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold, cross_val_predict, cross_validate
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier

import os
import socket

class LGBMClassifierCV(object):
    """cross_val_predict"""

    def __init__(self, params=None, cv=5, random_state=None, n_repeats=None):
        self.clf = LGBMClassifier()
        if params:
            self.clf.set_params(**params)
        if n_repeats:
            self._kf = RepeatedStratifiedKFold(cv, True, random_state) # 复制N次
            self._num_preds = cv * n_repeats
        else:
            self._kf = StratifiedKFold(cv, True, random_state)
            self._num_preds = cv

    def fit(self, X, y, X_test=None,
            feval=roc_auc_score,
            sample_weight=None,
            init_score=None,
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=100,
            feature_name='auto',
            categorical_feature='auto',
            callbacks=None):

        """输入数组"""
        if X_test is None:
            X_test = X[:1] # 将第一行作为test集

        self.oof_train = np.zeros(len(X))
        self.oof_test = np.zeros((len(X_test), self._num_preds)) # num_preds:有多少折

        for n_fold, (train_index, valid_index) in enumerate(self._kf.split(X, y)):
            if verbose:
                print("\033[94mFold %s started at %s\033[0m" % (n_fold + 1, time.ctime()))
            X_train, y_train = X[train_index], y[train_index]
            X_valid, y_valid = X[valid_index], y[valid_index]
            eval_set = [(X_train, y_train), (X_valid, y_valid)] # 需要同时验证两个集合

            ########################################################################
            self.clf.fit(X_train,
                         y_train,
                         sample_weight,
                         init_score,
                         eval_set,
                         eval_names=('Train', 'Valid'),
                         eval_sample_weight=None,
                         eval_class_weight=None,
                         eval_init_score=None,
                         eval_metric=eval_metric,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose=verbose,
                         feature_name=feature_name,
                         categorical_feature=categorical_feature,
                         callbacks=callbacks)

            self.oof_train[valid_index] = self.clf.predict_proba(X_valid)[:, 1]
            self.oof_test[:, n_fold] = self.clf.predict_proba(X_test)[:, 1]
            ########################################################################

        # 输出 测试集 out-of-fold
        self.oof_test_rank = (pd.DataFrame(self.oof_test).rank().mean(axis=1)/len(self.oof_test)).values
        self.oof_test = self.oof_test.mean(axis=1) # 测试集的oof score算平均

        assert len(X) == len(self.oof_train)
        assert len(X_test) == len(self.oof_test)

        # 计算 训练集 oof 得分(out_of_fold)
        if feval:
            self.oof_train_score = feval(y, self.oof_train)
            print(f"\n\033[94mtrain CV Score: {self.oof_train_score} ended at {time.ctime()}\033[0m")
            return self.oof_train_score

    def oof_submit(self, ids, pred_ranking=False, file=None, preds=None):
        """preds分用于submit"""
        if file is None:
            file = f'submit_{self.oof_train_score}.csv'
        print(f'Save {file} ...')

        if preds is None:
            preds = self.oof_test if pred_ranking else self.oof_test_rank

        if not isinstance(ids, pd.DataFrame):
            ids = pd.DataFrame(ids)
        ids.assign(preds=preds).to_csv(file, index=False, header=False)

    @property
    def oof_train_and_test(self):
        return np.r_[self.oof_train, self.oof_test]

    def oof_save(self, file='./oof_train_and_test.csv'):
        pd.DataFrame(self.oof_train_and_test, columns=['oof_train_and_test']).to_csv(file, index=False)

    def plot_feature_importances(self, feature_names=None, topk=20, figsize=(10, 6), pic_name=None):
        columns = ['Importances', 'Features']
        importances = self.clf.feature_importances_.tolist()
        if feature_names is None:
            feature_names = list(map(lambda x: f'F_{x}', range(len(importances))))
        _ = list(zip(importances, feature_names))
        df = pd.DataFrame(_, columns=columns).sort_values('Importances', 0, False)

        plt.figure(figsize=figsize)
        sns.barplot(*columns, data=df[:topk])
        plt.title('Features Importances\n')
        plt.tight_layout()
        if pic_name is None:
            plt.savefig(f'importances_{self.oof_train_score}.png')


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification()
    X_test, _ = make_classification()

    clf = LGBMClassifierCV()
    clf.fit(X, y, X_test)
    clf.plot_feature_importances()
    """
    一组lightgbmcv参数:
    params = {
 'class_weight':'balanced',
 'metric': 'auc',
 'boosting_type': 'gbdt',
 'objective': 'binary',
 'max_depth': -1,
 'num_leaves': 16,
 'learning_rate': 0.005,
 'min_split_gain': 0.884,
 'min_child_weight': 0.01,
 'min_child_samples': 31,
 'subsample': 0.788,
 'subsample_freq': 8,
 'colsample_bytree': 0.617,
 'reg_alpha': 0.631,
 'reg_lambda': 0.81,
 'scale_pos_weight': 1,
 'random_state': 666,
 'verbosity': -1,
 'n_jobs': -1,
 'n_estimators': 30000} # 300分数好像很高
 
    oof8 = LGBMClassifierCV(params, 8, 999)
    oof8.fit(X, y, X_test, early_stopping_rounds=300)
    """