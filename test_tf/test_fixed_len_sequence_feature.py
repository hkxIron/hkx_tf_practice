#coding=utf-8

import tensorflow as tf
import os

def t1():
    #keys=[[1.0,2.0],[2.0,3.0]]
    print("t1"+"="*20)
    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    def make_example(locale,age,score,times):
        example = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                "locale":tf.train.Feature(bytes_list=tf.train.BytesList(value=[locale])),
                "age":tf.train.Feature(int64_list=tf.train.Int64List(value=[age]))
            }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                "movie_rating":tf.train.FeatureList(feature=[tf.train.Feature(float_list=tf.train.FloatList(value=score)) for i in range(times)])
                }
            )
        )
        return example.SerializeToString()

    context_features = {
        "locale": tf.FixedLenFeature([],dtype=tf.string),
        "age": tf.FixedLenFeature([],dtype=tf.int64)
    }
    sequence_features = {
        "movie_rating": tf.FixedLenSequenceFeature([3], dtype=tf.float32,allow_missing=True)
    }

    context_parsed, sequence_parsed  = tf.parse_single_sequence_example(
        make_example(locale='china',age=24,score=[1.0,3.5,4.0],times=2),
        context_features=context_features,
        sequence_features=sequence_features)

    print(tf.contrib.learn.run_n(context_parsed))
    print(tf.contrib.learn.run_n(sequence_parsed))

    """
    [{'locale': 'china', 'age': 24}]

    [{'movie_rating': array([[ 1. ,  3.5,  4. ],
           [ 1. ,  3.5,  4. ]], dtype=float32)}]
    """

def t2():
    # coding=utf-8
    print("t2"+"="*20)
    import tensorflow as tf
    import os
    keys = [[1, 2], [2]]
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    def make_example(key):

        example = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(key)]))
                }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    "index": tf.train.FeatureList(
                        feature=[tf.train.Feature(int64_list=tf.train.Int64List(value=[key[i]])) for i in
                                 range(len(key))])
                }
            )
        )
        return example.SerializeToString()
    filename = "tmp.tfrecords"
    if os.path.exists(filename):
        os.remove(filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for key in keys:
        ex = make_example(key)
        writer.write(ex)
    writer.close()

    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(["tmp.tfrecords"], num_epochs=1)
    _, serialized_example = reader.read(filename_queue)

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    context_features = {
        "length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        #  The entries in the `batch` from different `Examples` will be padded with
        # `default_value` to the maximum length present in the `batch`.
        #  FixedLenSequenceFeature会与样本中最长的样本对齐，其它的全都padding成0
        "index": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    batch_data = tf.train.batch(tensors=[sequence_parsed['index']], batch_size=2, dynamic_pad=True) # 会以最长的那个样本进行pad
    result = tf.contrib.learn.run_n({"index": batch_data})
    print(result)
    """
    [{'index': array([[1, 2],
       [2, 0]], dtype=int64)}]
    """

#t1()
t2()