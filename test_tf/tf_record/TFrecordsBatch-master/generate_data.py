#coding=utf-8
"""
author:luchi
date:24/4/2017
desc:generate data using for logistic regression
"""
import numpy as np
from sklearn.datasets.samples_generator import make_classification
import tensorflow as tf
import matplotlib.pyplot as plt


def generate_data(data_num,data_dim):

    X1, Y1 = make_classification(n_samples=data_num, n_features=data_dim, n_redundant=0,n_clusters_per_class=1, n_classes=2)
    # plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
    # plt.show()
    return X1,Y1

def make_exmaple(features,label):
    ex = tf.train.Example(
        features=tf.train.Features(
            feature = {
                "data":tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
        )
    )
    return ex

def generate_tfrecords(data_num,data_dim,filename):
    X,Y = generate_data(data_num,data_dim)

    writer =tf.python_io.TFRecordWriter(filename)

    for x,y in zip(X,Y):
        ex = make_exmaple(x,y)
        writer.write(ex.SerializeToString())
    writer.close()

if __name__ == "__main__":
    generate_tfrecords(1000,2,"reg.tfrecords")

