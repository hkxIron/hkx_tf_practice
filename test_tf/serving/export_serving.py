import os
import sys
import shutil
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_integer('training_iteration', 1, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('export_path', 'export_base', '')
tf.app.flags.DEFINE_string('data_dir', 'D:\\public_code\\hkx_tf_practice\\data\\mnist-tf\\', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype='float')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, dtype='float')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    # if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    #     print('Usage: mnist_export.py [--training_iteration=x] '
    #           '[--model_version=y] export_dir')
    #     sys.exit(-1)
    if FLAGS.training_iteration <= 0:
        print('Please specify a positive value for training iteration.')
        sys.exit(-1)
    if FLAGS.model_version <= 0:
        print('Please specify a positive value for version number.')
        sys.exit(-1)

    print('Training...')
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    sess = tf.InteractiveSession()
    serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),} # 28*28 =784
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)

    x = tf.identity(tf_example['x'], name='x')
    y_ = tf.placeholder('float', shape=[None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    y = tf.nn.softmax(y_conv, name='y')
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    values, indices = tf.nn.top_k(y_conv, 10) # 从小到大排名每个分数以及相应的index
    prediction_classes = tf.contrib.lookup.index_to_string(
        tf.to_int64(indices),
        mapping=tf.constant([str(i) for i in range(10)]))

    sess.run(tf.global_variables_initializer())

    for _ in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0],
                                  y_: batch[1], keep_prob: 0.5})
        correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                                      tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('training accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels, keep_prob: 1.0}))

    print('training is finished!')

    export_path_base = FLAGS.export_path #"export_base" #sys.argv[-1]
    export_path = os.path.join(compat.as_bytes(export_path_base),compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    if os.path.exists(export_path_base):
        shutil.rmtree(export_path_base,ignore_errors=True)
    builder = saved_model_builder.SavedModelBuilder(export_path)


    # 分类的签名
    classification_inputs = utils.build_tensor_info(serialized_tf_example)
    classification_outputs_classes = utils.build_tensor_info(prediction_classes) # 预测的top 10的类别
    classification_outputs_scores = utils.build_tensor_info(values) # 预测的top 10的分数
    classification_signature = signature_def_utils.build_signature_def(
        inputs={signature_constants.CLASSIFY_INPUTS:classification_inputs},
        # 注意：此处的输出有两个值
        outputs={
            signature_constants.CLASSIFY_OUTPUT_CLASSES:
                classification_outputs_classes,
            signature_constants.CLASSIFY_OUTPUT_SCORES:
                classification_outputs_scores
        },
        method_name=signature_constants.CLASSIFY_METHOD_NAME
    )

    # 预测的签名
    tensor_info_x = utils.build_tensor_info(x)
    tensor_info_y = utils.build_tensor_info(y_conv) #预测时不需要再计算softmax，直接比较就行
    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'images': tensor_info_x},
        outputs={'scores': tensor_info_y}, # 预测成为不同label的分数
        method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            'predict_images': prediction_signature,
            "class_signature": classification_signature,
            #signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature, # serving_default
        },
        legacy_init_op=legacy_init_op)

    builder.save()
    print('new model exported!')

if __name__ == '__main__':
    tf.app.run()