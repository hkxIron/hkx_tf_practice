import tensorflow as tf

# g1 中设其值为0
g1 =tf.Graph()
with g1.as_default():
    v1 = tf.get_variable("v",[1,1],initializer=tf.zeros_initializer)

# g2 中设其值为1
g2 =tf.Graph()
with g2.as_default():
    v2 = tf.get_variable("v",[1,1],initializer=tf.ones_initializer)


#在计算图g1中读取变量"v"的取值
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中，变量"v"的取值为0,输出为0
        print("print g1 v:",sess.run(tf.get_variable("v")))

# 在计算图g2中读取变量"v"的取值
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g1中，变量"v"的取值为1,输出为1
        print("print g2 v:",sess.run(tf.get_variable("v")))

