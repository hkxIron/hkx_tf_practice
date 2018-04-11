#coding=utf-8
"""
author:luchi
date:24/4/2017
desc:logistic regression
"""
import tensorflow as tf
class Logistic(object):

    def __init__(self,config,data,label):

        self.data =data
        self.label =label
        data_dim = config.data_dim
        label_dim = config.label_num
        lr = config.learining_rate


        softmax_w = tf.get_variable(name="softmax_w",shape=[data_dim,label_dim])
        softmax_b =tf.get_variable(name="softmax_b",shape=[label_dim])




        with tf.name_scope("logist"):
            self.logits=tf.matmul(self.data,softmax_w)+softmax_b

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10,labels=self.label))


        self.prediction = tf.arg_max(self.logits,1)
        self.correct =  tf.cast(tf.equal(self.prediction,self.label),tf.float32)

        self.accuracy = tf.reduce_mean(self.correct)

        optimizer = tf.train.GradientDescentOptimizer(lr)

        self.train_op = optimizer.minimize(loss=self.loss)



