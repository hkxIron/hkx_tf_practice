import tensorflow as tf
import os,sys,shutil

"""
本方法的优势：其实已经保存了结构信息，获取数据时不需要构建一遍data graph。
本方法的劣势：比如知道输入、输出变量的名称。如果有多个模型实现同一个功能，如果模型的输入名称不同，则不能统一接口。
"""


save_path = "saver/builder_graph/"
def save_with_graph():
    print("save with graph")
    x1 = tf.placeholder(tf.float32, name='x1')
    x2 = tf.get_variable('x2', initializer=tf.constant(1.0), dtype=tf.float32)
    y = x1 + x2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(y.eval(feed_dict={x1: 15}))
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)  # 该文件夹不能存在
        builder.add_meta_graph_and_variables(sess, ['tag_string'])
        builder.save()


def load_with_graph():
    print("load with graph")
    # load(包括结构)
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, ['tag_string'], save_path)
        x1 = sess.graph.get_tensor_by_name('x1:0')
        x2 = sess.graph.get_tensor_by_name('x2:0')
        y = sess.graph.get_tensor_by_name('add:0')
        print(sess.run(y, feed_dict={x1: 25}))


save_with_graph()
load_with_graph()