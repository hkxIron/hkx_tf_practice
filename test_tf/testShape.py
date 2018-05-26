import tensorflow as tf

# case 2
input = tf.Variable(tf.random_normal([1, 3, 3, 5])) # NHWC
filter = tf.Variable(tf.random_normal([1, 1, 5, 1])) # filter_h,filter_w,in_c,out_c
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID') # out_n:1,out_h:(3-1+1)/1=3, out_width:(3-1+1)/1,out_c:1
# op: 1*3*3*1
test = tf.constant([[2,3],[-1,-2]])
yy = test>tf.constant(0)
zz = tf.cast(yy,dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = (sess.run(op))
    print("shape1:",res.shape)
    print("yy:",sess.run(yy))
    print("zz:",sess.run(zz))


input = tf.Variable(tf.random_normal([1, 5, 5, 5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    res = (sess.run(op))
    print("shape2:",res.shape)

