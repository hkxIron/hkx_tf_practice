#coding:utf-8
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter('ignore')
#print(" cross valid")

def test1():
    dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=",")

    X = dataset[:,0:8]
    Y = dataset[:,8]
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    #监控模型表现
    model = XGBClassifier()
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

    from xgboost import plot_importance
    from matplotlib import pyplot

    model.fit(X, Y)

    plot_importance(model)
    pyplot.show()

    # 交叉验证
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedKFold
    model = XGBClassifier()
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    # 搜索到最好的参数
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def test2():
    from numpy import loadtxt
    from xgboost import XGBClassifier
    from matplotlib import pyplot
    # load data
    dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=",")
    # split data into X and y
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    # fit model no training data
    model = XGBClassifier()
    model.fit(X, y)
    # feature importance
    print(model.feature_importances_)
    # plot
    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()

def test3():
    from numpy import loadtxt
    from xgboost import XGBClassifier
    from matplotlib import pyplot
    # load data
    dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=",")
    # split data into X and y
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    # fit model no training data
    model = XGBClassifier()
    model.fit(X, y)
    # feature importance
    print("feature importance:",model.feature_importances_)
    # plot
    #pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    from xgboost import plot_importance
    plot_importance(model)
    pyplot.show()

def test4():
    from numpy import loadtxt
    from numpy import sort
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.feature_selection import SelectFromModel
    # load data
    dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=",")
    # split data into X and y
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=7)
    # fit model on all training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # Fit model using each importance as a threshold
    thresholds = sort(model.feature_importances_)
    print("thresholds:", thresholds)
    for thresh in thresholds:
        # select features using threshold
        # features whose importance is greater or equal are kept while the others are discarded.
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
        """
        thresholds: [0.07094595 0.07263514 0.08445946 0.08952703 0.12837838 0.16047297 0.1858108  0.20777027]
        
        Thresh=0.071, n=8, Accuracy: 77.95%
        Thresh=0.073, n=7, Accuracy: 76.38%
        Thresh=0.084, n=6, Accuracy: 77.56%
        Thresh=0.090, n=5, Accuracy: 76.38%
        Thresh=0.128, n=4, Accuracy: 76.38%
        Thresh=0.160, n=3, Accuracy: 74.80%
        Thresh=0.186, n=2, Accuracy: 71.65%
        Thresh=0.208, n=1, Accuracy: 63.78%
        """

#test1()
#test2()
#test3()
test4()
