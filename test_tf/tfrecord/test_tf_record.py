#!/usr/bin/env python
# encoding: utf-8
#
# 生成模拟数据

import tensorflow as tf
import os,sys

value_feat_id_name = "V"
value_feat_weight_name = "W"
index_feat_id_name = "X"
feat_bag_id_name = "b"
label_name = "y"
sum_abs_weight_name = "s"

_float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
_int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
def make_example(value_feat_id_list, value_feat_weight_list, index_feat_id_list, feat_bag_id, label):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                value_feat_id_name: _int_feature(value_feat_id_list),
                value_feat_weight_name: _float_feature(value_feat_weight_list),
                index_feat_id_name: _int_feature(index_feat_id_list),
                feat_bag_id_name: _int_feature(feat_bag_id),
                label_name: _int_feature(label)
            })
    )
    return example

def _decode_single_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            value_feat_id_name: tf.VarLenFeature(tf.int64),
            value_feat_weight_name: tf.VarLenFeature(tf.float32),
            index_feat_id_name: tf.VarLenFeature(tf.int64),
            feat_bag_id_name: tf.VarLenFeature(tf.int64),
            label_name: tf.FixedLenFeature([], tf.int64,default_value=0), # 如果是FixedLenFeature的话，读取来就是ndarray,如果是VarLenFeature的话，就是SparseTensor
        })
    # 下面的必须要保证所有的Tensor都不能为空
    # value_feat_ids,_ =tf.sparse_fill_empty_rows(features[value_feat_id_name],default_value=10 ) # sparseTensor
    # value_feat_weights,_ =tf.sparse_fill_empty_rows(features[value_feat_weight_name],default_value=0 ) # sparseTensor
    # index_feat_ids,_ =tf.sparse_fill_empty_rows(features[index_feat_id_name],default_value=11)  # sparseTensor
    value_feat_ids = features[value_feat_id_name] # sparseTensor
    value_feat_weights = features[value_feat_weight_name] # sparseTensor
    index_feat_ids = features[index_feat_id_name]  # sparseTensor
    bag_id = features[feat_bag_id_name]  # ndarray
    label = features[label_name]  # ndarray
    return value_feat_ids, value_feat_weights, index_feat_ids, bag_id, label

def _batch_inputs(files, batch_size, num_epochs, num_preprocess_threads):
    """Reads input data num_epochs times.
    """
    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        single_example = _decode_single_example(serialized_example)
        #example_list =[ decode_single_example(serialized_example) for _ in range(10)]
        #batch_examples = tf.train.shuffle_batch(
        batch_examples = tf.train.batch(
            list(single_example),
            #example_list,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=1000 + 3 * batch_size,
            allow_smaller_final_batch=True
            # Ensures a minimum amount of shuffling of examples.
            #min_after_dequeue=min_after_dequeue
        )
        return batch_examples

def write_record(file_name, dataset):
    if os.path.exists(file_name): os.remove(file_name)
    writer = tf.python_io.TFRecordWriter(file_name)
    for data in dataset:
        ex = make_example(data[value_feat_id_name], data[value_feat_weight_name], data[index_feat_id_name],
                          [data[feat_bag_id_name]], [data[label_name]]).SerializeToString()
        writer.write(ex)
    writer.close()

def main():
    filename = "../../../data/tmp.tfrecords"
    dataset = [
        {
         value_feat_id_name: [2, 3], value_feat_weight_name: [0.5, 0.3], index_feat_id_name: [1, 5],
         feat_bag_id_name: 10, label_name: 1
        },
        {
            value_feat_id_name: [2, 4], value_feat_weight_name: [0.3, 0.3], index_feat_id_name: [1, 7],
            feat_bag_id_name: 11, label_name: 0
        },
        {
            value_feat_id_name: [2, 4], value_feat_weight_name: [0.3, 0.2], index_feat_id_name: [5, 7],
            feat_bag_id_name: 13, label_name: 1
        },
        {
            value_feat_id_name: [4], value_feat_weight_name: [0.3], index_feat_id_name: [5, 7],
            feat_bag_id_name: 14, label_name: 0
        },
        {value_feat_id_name: [6, 7], value_feat_weight_name: [0.5, 0.2], index_feat_id_name: [], feat_bag_id_name: 15,
         label_name: 1} #,
        # {value_feat_id_name: [], value_feat_weight_name: [], index_feat_id_name: [5], feat_bag_id_name: 10,
        #   label_name: 0}
    ]

    print("begin write!")
    #write_record(filename, dataset)
    print("begin read!")
    with tf.Graph().as_default():
        sess = tf.InteractiveSession()
        batch_size = 1
        batch_data = _batch_inputs([filename], batch_size,num_epochs=1,num_preprocess_threads=1)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while not coord.should_stop() and step < len(dataset) / batch_size:
                value_feat_ids, value_feat_weights, index_feat_ids, bag_id, label = sess.run(list(batch_data))
                print("value_feat_ids:", value_feat_ids, "\nvalue_feat_weights:", value_feat_weights, "\nindex_feat_ids:",
                index_feat_ids, "\nbag_id:", bag_id, "\nlabel:", label, "label type:",type(label),"\n--------------------\n")
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done training .')
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()
if __name__ == '__main__':
    main()