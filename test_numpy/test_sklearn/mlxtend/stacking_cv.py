from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

def fun1():
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import StackingCVClassifier
    import numpy as np
    import warnings

    warnings.simplefilter('ignore')

    RANDOM_SEED = 42

    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=RANDOM_SEED)
    clf3 = GaussianNB()
    lr = LogisticRegression()

    # Starting from v0.16.0, StackingCVRegressor supports
    # `random_state` to get deterministic result.
    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                                meta_classifier=lr,
                                random_state=RANDOM_SEED)

    print('3-fold cross validation:\n')

    for clf, label in zip([clf1, clf2, clf3, sclf],
                          ['KNN',
                           'Random Forest',
                           'Naive Bayes',
                           'StackingClassifier']):

        scores = model_selection.cross_val_score(clf, X, y,
                                                  cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))


    import matplotlib.pyplot as plt
    from mlxtend.plotting import plot_decision_regions
    import matplotlib.gridspec as gridspec
    import itertools

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10, 8))

    for clf, lab, grd in zip([clf1, clf2, clf3, sclf],
                             ['KNN',
                              'Random Forest',
                              'Naive Bayes',
                              'StackingCVClassifier'],
                             itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf)
        plt.title(lab)
    plt.show()

def fun2():
    from sklearn import model_selection
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import StackingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from mlxtend.classifier import StackingClassifier
    from mlxtend.classifier import StackingCVClassifier
    import numpy as np
    import warnings
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()

    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                                use_probas=True,
                                meta_classifier=lr,
                                random_state=42)

    print('3-fold cross validation:\n')

    for clf, label in zip([clf1, clf2, clf3, sclf],
                          ['KNN',
                           'Random Forest',
                           'Naive Bayes',
                           'StackingClassifier']):
        scores = model_selection.cross_val_score(clf, X, y,
                                                 cv=3, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))

#fun1()
fun2()