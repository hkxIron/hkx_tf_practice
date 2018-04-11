#!/usr/bin/env python
# coding:utf-8
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige
#          \date   2016-07-19 17:09:07.466651
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, time
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')

MIN_AFTER_DEQUEUE = 10000


def read(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    return serialized_example


def decode(batch_serialized_examples):
    features = tf.parse_example(
        batch_serialized_examples,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'index': tf.VarLenFeature(tf.int64),
            'value': tf.VarLenFeature(tf.float32),
        })

    label = features['label']
    index = features['index']
    value = features['value']

    return label, index, value


def batch_inputs(files, batch_size, num_epochs=None, num_preprocess_threads=1):
    """Reads input data num_epochs times.
    """
    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)

        serialized_example = read(filename_queue)
        # here we for simple we will only show using shuffle_batch, you can ref to shuffle_batch_join, and batch
        # I will add test code showing the diff
        batch_serialized_examples = tf.train.shuffle_batch(
            [serialized_example],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=MIN_AFTER_DEQUEUE + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=MIN_AFTER_DEQUEUE)

        return decode(batch_serialized_examples)


def read_records():
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input images and labels.
        tf_record_pattern ="data/output"  #sys.argv[1]
        data_files = tf.gfile.Glob(tf_record_pattern)
        label, index, value = batch_inputs(
            data_files,
            batch_size=FLAGS.batch_size,
            num_epochs=FLAGS.num_epochs,
            num_preprocess_threads=FLAGS.num_preprocess_threads)

        # The op for initializing the variables.
        # notice you need to use initialize_local_variables other wise if you set num_epochs not None will fail
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.InteractiveSession()
        # Initialize the variables (the trained variables and the
        # epoch counter).
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                label_, index_, value_ = sess.run([label, index, value])
                if step == 0:
                    print('--label show dense tensor result')
                    print(label_)
                    print(len(label_))
                    print('--show SpareTesnor index_ and value_')
                    print(index_)
                    print(value_)
                    print('--index[0] as indices')
                    # --indices
                    print(index_[0])
                    # --values
                    print('--index[1] as values')
                    print(index_[1])
                    # --shape
                    print('--index[2] as sparse tensor shape')
                    print(index_[2])
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        sess.close()


def main(_):
    read_records()


if __name__ == '__main__':
    tf.app.run()