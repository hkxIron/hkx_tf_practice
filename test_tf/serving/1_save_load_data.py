# blog:https://www.zybuluo.com/irving512/note/962531
import tensorflow as tf
import shutil
import os
# save操作其实保存的不只是变量数据，还包括了结构，但一般只用到变量数据。
save_path="save/"
def save():
    print("save...")
    x1 = tf.placeholder(tf.float32)
    x2 = tf.get_variable('x2', initializer=tf.constant(10.0), dtype=tf.float32)
    y = x1 + x2
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(y.eval(feed_dict={x1: 1}))
        saver.save(sess,os.path.join(save_path,"model.ckpt"))

def load():
    print("load...")
    x1 = tf.placeholder(tf.float32)
    with tf.variable_scope(tf.get_variable_scope(),reuse=True): # reuse必须要加，否则会出错
        x2 = tf.get_variable('x2', initializer=tf.constant(1.0), dtype=tf.float32)
    y = x1 + x2
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 获取x2的数据
        saver.restore(sess, os.path.join(save_path,"model.ckpt"))
        print(y.eval(feed_dict={x1: 7}))

save()
load()

