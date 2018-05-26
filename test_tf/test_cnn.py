import numpy as np
import tensorflow as tf

print("valid")
input = tf.Variable(tf.random_normal([1,3,3,5])) # [batch, in_height, in_width, in_channels]
filter = tf.Variable(tf.random_normal([1,1,5,1])) # [filter_height, filter_width, in_channels, out_channels]
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')  #strides=[1, in_height, in_width, 1]在图像各维上的步长,其中strides[0] = strides[3] = 1为固定值,不可改变.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(op)
    print(res.shape)  # out_height=out_width=(3 - 1)/1+1=3

print("valid")
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(op)
    print(res.shape) # out_height=out_width= (5-3)/1+1=3

print("same")
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 2, 1], padding='SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(op)
    print(res.shape) # out_height=ceil(5/1)=5,out_width = ceil(5/2)=3

print("same")
input = tf.Variable(tf.random_normal([1,6,17,1]))
filter = tf.Variable(tf.random_normal([3,5,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 2, 4, 1], padding='SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = sess.run(op)
    print(res.shape) # out_height= ceil(6/2)=3, out_width=ceil(17/4)=5
# pad : (ceil(img/stride)-1)*stride+filter - img = (img-1)/stride*stride+filter-img
