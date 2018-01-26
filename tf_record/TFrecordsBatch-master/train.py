#coding=utf-8
"""
author:luchi
date:24/4/2017
desc:training logistic regression
"""
import tensorflow as tf
from model import Logistic

def read_my_file_format(filename_queue):
    reader = tf.TFRecordReader()
    _,serilized_example = reader.read(filename_queue)

    #parsing example
    features = tf.parse_single_example(serilized_example,
        features={
            "data":tf.FixedLenFeature([2],tf.float32),
            "label":tf.FixedLenFeature([],tf.int64)
        }

    )

    #decode from raw data,there indeed do not to change ,but to show common step , i write a case here

    # data = tf.cast(features['data'],tf.float32)
    # label = tf.cast(features['label'],tf.int64)

    return features['data'],features['label']


def input_pipeline(filenames, batch_size, num_epochs=100):

    filename_queue = tf.train.string_input_producer([filenames],num_epochs=num_epochs)
    data,label=read_my_file_format(filename_queue)

    datas,labels = tf.train.shuffle_batch([data,label],batch_size=batch_size,num_threads=5,
                                          capacity=1000+3*batch_size,min_after_dequeue=1000)
    return datas,labels

class config():
    data_dim=2
    label_num=2
    learining_rate=0.1
    init_scale=0.01

def run_training():

    with tf.Graph().as_default(), tf.Session() as sess:

        datas,labels = input_pipeline("reg.tfrecords",32)

        c = config()
        initializer = tf.random_uniform_initializer(-1*c.init_scale,1*c.init_scale)

        with tf.variable_scope("model",initializer=initializer):
            model = Logistic(config=c,data=datas,label=labels)

        fetches = [model.train_op,model.accuracy,model.loss]
        feed_dict={}

        #init
        init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            while not coord.should_stop():

                # fetches = [model.train_op,model.accuracy,model.loss]
                # feed_dict={}
                # feed_dict[model.data]=sess.run(datas)
                # feed_dict[model.label]=sess.run(labels)
                # _,accuracy,loss= sess.run(fetches,feed_dict)
                _,accuracy,loss= sess.run(fetches,feed_dict)
                print("the loss is %f and the accuracy is %f"%(loss,accuracy))
        except tf.errors.OutOfRangeError:
            print("done training")
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

def main():
    run_training()

if __name__=='__main__':
    main()
