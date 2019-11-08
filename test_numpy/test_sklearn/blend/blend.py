"""Kaggle competition: Predicting a Biological Response.
Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/10950#post10950
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)
The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)
Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.
Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.

手工blend
"""

from __future__ import division
import numpy as np
import load_data
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss, i.e. the score of the bioresponse competition.
    """
    attempt = np.clip(attempt, epsilon, 1.0 - epsilon)
    return - np.mean(actual * np.log(attempt) +
                     (1.0 - actual) * np.log(1.0 - attempt))

if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set
    n_folds = 10
    verbose = True
    shuffle = False
    #X, y, X_submission = load_data.load()
    X, y, X_submission = load_data.generate_data()
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds)) # 按照类别分层采样
    # 训练K个模型
    models = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
              RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
              ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
              ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
              GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)
              ]

    print("Creating train and test sets for blending.")
    dataset_blend_train = np.zeros((X.shape[0], len(models))) # [N, K]: 每个模型的预测为其中一行
    dataset_blend_test = np.zeros((X_submission.shape[0], len(models))) # [N, K]

    train_auc = []
    for j, model in enumerate(models):
        print("model index:", j, " clf:", model)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf))) #[N,k]
        for i, (train_index, test_index) in enumerate(skf):
            print("fold:", i)
            #print("fold:", i, " train size:", train_index.shape, " test size:", test_index.shape)
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1] # 只得到概率, [N,1]
            dataset_blend_train[test_index, j] = y_pred # 第j个模型的预测放在第j列
            # 最终提交的预测
            dataset_blend_test_j[:, i] = model.predict_proba(X_submission)[:, 1] # out_of_fold预测

        # 计算模型auc
        y_pred_full = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, y_pred_full)
        print("full auc:{}".format(auc))
        train_auc.append(auc)
        # 对各列进行平均
        dataset_blend_test[:, j] = dataset_blend_test_j.mean(axis=1)

    print("max train auc:{}".format(max(train_auc))) # max train auc:0.9950696844598054
    # blending:用LR在所有基分类器的预测上训练
    # dataset_blend_train:[N, K]
    print("Blending.")
    #model = LogisticRegression()
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
    model.fit(dataset_blend_train, y)
    y_pred_blend = model.predict_proba(dataset_blend_train)[:, 1]
    print("blend auc:", metrics.roc_auc_score(y, y_pred_blend)) # auc=1, 要比blend之前高一些

    # pred submit
    y_submission = model.predict_proba(dataset_blend_test)[:, 1]
    print("Linear stretch of predictions to [0,1]")
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

    print("Saving Results.")
    tmp = np.vstack([range(1, len(y_submission) + 1), y_submission]).T
    np.savetxt(fname='submission.csv',
               X=tmp,
               fmt='%d,%0.9f',
               header='MoleculeId,PredictedProbability',
               comments='')