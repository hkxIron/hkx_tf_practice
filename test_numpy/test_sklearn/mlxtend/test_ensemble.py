from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import numpy as np
import warnings
warnings.simplefilter('ignore')

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1, 1, 1])

def test1():

    print('5-fold cross validation:\n')

    labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes']

    for clf, label in zip([clf1, clf2, clf3], labels):

        scores = model_selection.cross_val_score(clf, X, y,
                                                  cv=5,
                                                  scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))

def test2():


    labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
    for clf, label in zip([clf1, clf2, clf3, eclf], labels):
        scores = model_selection.cross_val_score(clf, X, y,
                                                 cv=5,
                                                 scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))

    """
    Accuracy: 0.90 (+/- 0.05) [Logistic Regression]
    Accuracy: 0.93 (+/- 0.05) [Random Forest]
    Accuracy: 0.91 (+/- 0.04) [Naive Bayes]
    Accuracy: 0.95 (+/- 0.05) [Ensemble]
    """

def test3():
    # Plotting Decision Regions
    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.gridspec as gridspec
    import itertools

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10, 8))

    labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
    for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                             labels,
                             itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf)
        plt.title(lab)

def test4():
    # Example 2 - Grid Search
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import EnsembleVoteClassifier

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')

    params = {'logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200], }

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))


def test5():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import EnsembleVoteClassifier

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    eclf = EnsembleVoteClassifier(clfs=[clf1, clf1, clf2],
                                  voting='soft')

    # If the EnsembleClassifier is initialized with multiple similar estimator objects, the estimator names are modified with consecutive integer indices, for example:
    params = {'logisticregression-1__C': [1.0, 100.0],
              'logisticregression-2__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200], }

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid = grid.fit(iris.data, iris.target)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

def test6():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import EnsembleVoteClassifier
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    eclf1 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='hard', verbose=1)
    eclf1 = eclf1.fit(X, y)
    print(eclf1.predict(X))
    eclf2 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    eclf2 = eclf2.fit(X, y)
    print(eclf2.predict(X))
    eclf3 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[2, 1, 1])
    eclf3 = eclf3.fit(X, y)
    print(eclf3.predict(X))

def test7():
    from sklearn import datasets

    iris = datasets.load_iris()
    X, y = iris.data[:, :], iris.target

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import EnsembleVoteClassifier
    from sklearn.pipeline import Pipeline
    from mlxtend.feature_selection import SequentialFeatureSelector

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()

    # Creating a feature-selection-classifier pipeline

    sfs1 = SequentialFeatureSelector(clf1,
                                     k_features=4,
                                     forward=True,
                                     floating=False,
                                     scoring='accuracy',
                                     verbose=0,
                                     cv=0)

    clf1_pipe = Pipeline([('sfs', sfs1),
                          ('logreg', clf1)])

    eclf = EnsembleVoteClassifier(clfs=[clf1_pipe, clf2, clf3],
                                  voting='soft')

    params = {'pipeline__sfs__k_features': [1, 2, 3],
              'pipeline__logreg__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

def test8():
    # Example 3 - Majority voting with classifiers trained on different feature subsets
    from sklearn import datasets

    iris = datasets.load_iris()
    X, y = iris.data[:, :], iris.target

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import EnsembleVoteClassifier
    from sklearn.pipeline import Pipeline
    from mlxtend.feature_selection import SequentialFeatureSelector

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()

    # Creating a feature-selection-classifier pipeline

    sfs1 = SequentialFeatureSelector(clf1,
                                     k_features=4,
                                     forward=True,
                                     floating=False,
                                     scoring='accuracy',
                                     verbose=0,
                                     cv=0)

    clf1_pipe = Pipeline([('sfs', sfs1),
                          ('logreg', clf1)])

    eclf = EnsembleVoteClassifier(clfs=[clf1_pipe, clf2, clf3],
                                  voting='soft')

    params = {'pipeline__sfs__k_features': [1, 2, 3],
              'pipeline__logreg__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    cv_keys = ('mean_test_score', 'std_test_score', 'params')

    print("test8")
    for r, _ in enumerate(grid.cv_results_['mean_test_score']):
        print("%0.3f +/- %0.2f %r"
              % (grid.cv_results_[cv_keys[0]][r],
                 grid.cv_results_[cv_keys[1]][r] / 2.0,
                 grid.cv_results_[cv_keys[2]][r]))

test8()
test1()
test2()
test3()
test4()
test5()
test6()
test7()


