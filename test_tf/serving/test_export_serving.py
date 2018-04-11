import os
import sys
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_integer('training_iteration', 10, 'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('export_path', 'export_base/1', 'Working directory.')
tf.app.flags.DEFINE_string('data_dir', 'D:\\public_code\\hkx_tf_practice\\data\\mnist-tf\\', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def predict():
    with tf.Session() as sess:
        meta_graph_def = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING], FLAGS.export_path)
        signature = meta_graph_def.signature_def

        # extract input/outputs
        signature_name = "class_signature"
        input_image_name = signature[signature_name].inputs[tf.saved_model.signature_constants.CLASSIFY_INPUTS].name
        output_image_name = signature[signature_name].outputs[tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES].name
        image_input = sess.graph.get_tensor_by_name(input_image_name)
        image_output = sess.graph.get_tensor_by_name(output_image_name)

        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32), }  # 28*28 =784
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        x = tf.identity(tf_example['x'], name='x')

        for _ in range(3):
            batch = mnist.train.next_batch(1)
            print("batch type:",type(batch))
            image = batch[0]
            [score_res]=sess.run([image_output], feed_dict={x:[image]})
            print("scores_res:", score_res)

def main(argv=None):
  predict()


if __name__ == '__main__':
  tf.app.run()
