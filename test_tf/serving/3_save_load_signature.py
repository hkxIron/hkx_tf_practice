import tensorflow as tf
import os,shutil

"""
如果不知道输入输出tensor的name，可以通过以下方法进行存取模型。
要先定义好输入/输出的name。
save
"""

save_path="saver/builder_signature/"
signature_key = 'test_signature'
model_tag = "test_saved_model"
input_key1 = 'x11'
input_key2 = 'x22'
output_key1 = 'y11'
output_key2 = 'y22'

def save():
    print("save")
    x1 = tf.placeholder(tf.float32, name='x1')
    x2 = tf.get_variable('x2', initializer=tf.constant(2.0), dtype=tf.float32)
    y = x1+x2
    y_mul = x1*x2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)
        # 指定输入变量名称为x11 x22
        # 指定输出变量名称为y11
        inputs = {input_key1: tf.saved_model.utils.build_tensor_info(x1),
                  input_key2: tf.saved_model.utils.build_tensor_info(x2)}
        outputs = {output_key1: tf.saved_model.utils.build_tensor_info(y),
                   output_key2: tf.saved_model.utils.build_tensor_info(y_mul)
                   }
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
                                                                           outputs= outputs,
                                                                           method_name='test_sig_name')
        builder.add_meta_graph_and_variables(sess, tags=[model_tag],
                                             signature_def_map={signature_key: signature}
                                             )
        builder.save()

def load():
    print("load")
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, tags=[model_tag],export_dir =save_path)
        # 从meta_graph_def中取出SignatureDef对象
        signature = meta_graph_def.signature_def
        # 从signature中找出具体输入输出的tensor name
        x1 = signature[signature_key].inputs[input_key1].name
        x2 = signature[signature_key].inputs[input_key2].name
        y1 = signature[signature_key].outputs[output_key1].name
        y2 = signature[signature_key].outputs[output_key2].name
        print("x1 name:",x1) # x1:0
        print("x2 name:",x2) # x2:0
        # 获取tensor 并inference
        x11 = sess.graph.get_tensor_by_name(x1) # 4
        x22 = sess.graph.get_tensor_by_name(x2) # 2
        y11 = sess.graph.get_tensor_by_name(y1)
        y22 = sess.graph.get_tensor_by_name(y2)
        # _x 实际输入待inference的data
        print(sess.run([y11,y22], feed_dict={x11: 4}))

save()
load()