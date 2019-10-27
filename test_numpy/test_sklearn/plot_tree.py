#-*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus
import os
#os.environ["PATH"] += os.pathsep + 'G:/program_files/graphviz/bin'


def decision_tree():
    # -*- coding: utf-8 -*-
    from itertools import product

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier

    from IPython.display import Image
    from sklearn import tree
    import pydotplus
    import os

    # 仍然使用自带的iris数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 训练模型，限制树的最大深度4
    clf = DecisionTreeClassifier(max_depth=4, random_state=0)
    # 拟合模型
    clf.fit(X, y)

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    Image(graph.create_png())
    # 如果没有ipython的jupyter notebook，可以把此图写到pdf文件里，在pdf文件里查看。
    graph.write_pdf("iris_dt.pdf")

def random_forest():
    # 仍然使用自带的iris数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 训练模型，限制树的最大深度4
    clf = RandomForestClassifier(max_depth=4, random_state=0)
    #拟合模型
    clf.fit(X, y)

    #Estimators = clf.estimators_
    for index, model in enumerate(clf.estimators_):
        filename = 'rf_iris_' + str(index) + '.pdf'
        dot_data = tree.export_graphviz(model, out_file=None,
                             feature_names=iris.feature_names,
                             class_names=iris.target_names,
                             filled=True, rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # 使用ipython的终端jupyter notebook显示。
        Image(graph.create_png())
        if index<=2:
            graph.write_pdf(filename)


def gbdt_tree():
    from sklearn.ensemble import GradientBoostingClassifier
    # 仍然使用自带的iris数据
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 训练模型，限制树的最大深度4
    clf = GradientBoostingClassifier(max_depth=4, random_state=0)
    #拟合模型
    clf.fit(X, y)

    #Estimators = clf.estimators_
    for index, model in enumerate(clf.estimators_):
        print("model size:", len(model)) # 4分类,每次会生成3颗树
        for class_index in range(len(model)):
            filename = 'gbdt_iris_' + str(index) +"_class"+str(class_index)+ '.pdf'
            dot_data = tree.export_graphviz(model[class_index],
                                            out_file=None,
                                            feature_names=iris.feature_names,
                                            class_names=iris.target_names,
                                            filled=True, rounded=True,
                                            special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            # 使用ipython的终端jupyter notebook显示。
            Image(graph.create_png())
            graph.write_pdf(filename)
            if index>1:# 只画前两个
                break


#decision_tree()
#random_forest()
gbdt_tree()