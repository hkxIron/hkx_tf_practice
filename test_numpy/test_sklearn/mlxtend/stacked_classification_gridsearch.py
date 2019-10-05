from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingClassifier

from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

def stack1():
    # Initializing models
    print("X:{} y:{}".format(X.shape, y.shape))

    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              meta_classifier=lr)

    params = {'kneighborsclassifier__n_neighbors': [1, 5], # 最近邻的参数
              'randomforestclassifier__n_estimators': [10, 50], # 随机森林的参数
              'meta_classifier__C': [0.1, 10.0]} # lr的正则项的倒数

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)

def stack2():
    from sklearn.model_selection import GridSearchCV

    # Initializing models

    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf1, clf2, clf3],
                              meta_classifier=lr)

    params = {'kneighborsclassifier-1__n_neighbors': [1, 5],
              'kneighborsclassifier-2__n_neighbors': [1, 5],
              'randomforestclassifier__n_estimators': [10, 50],
              'meta_classifier__C': [0.1, 10.0]}

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

    print('Best parameters: %s' % grid.best_params_)
    print('Accuracy: %.2f' % grid.best_score_)

stack1()
stack2()
