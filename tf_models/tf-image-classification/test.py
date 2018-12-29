import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
import hickle as hkl
import os.path as osp
from glob import glob
import sklearn.metrics as metrics

from input import Dataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model
import globals as g_ 
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', 'tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '', 
                            """finetune with a pretrained model""")
tf.app.flags.DEFINE_string('caffemodel', '', 
                            """finetune with a model converted by caffe-tensorflow""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.05  # Learning rate decay factor.

np.set_printoptions(precision=3)


def test(dataset, ckptfile):
    print 'train() called'
    batch_size = FLAGS.batch_size

    data_size = dataset.size()
    print 'training size:', data_size


    with tf.Graph().as_default():
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)
         
        image_, y_ = model.input()
        keep_prob_ = tf.placeholder('float32', name='keep_prob')
        phase_train_ = tf.placeholder(tf.bool, name='phase_train')

        logits = model.inference(image_, keep_prob_, phase_train_) 
        prediction = model.classify(logits)
        loss, print_op = model.loss(logits, y_)
        train_op = model.train(loss, global_step, data_size)


        # build the summary operation based on the F colection of Summaries
        summary_op = tf.merge_all_summaries()

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        if FLAGS.caffemodel:
            caffemodel = FLAGS.caffemodel
            # sess.run(init_op)
            model.load_model(sess, caffemodel, fc8=True)
            print 'loaded pretrained caffemodel:', caffemodel
        else:
            saver.restore(sess, ckptfile)
            print 'restore variables done'

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def) 

        step = startstep
            
        predictions = []
        labels = []

        for batch_x, batch_y in dataset.batches(batch_size):
            if step >= FLAGS.max_steps:
                break
            step += 1

            
            if step == 1:
                img = batch_x[0,...]
                cv2.imwrite('img0.jpg', img)


            start_time = time.time()
            feed_dict = {image_: batch_x,
                         y_ : batch_y,
                         keep_prob_: 1.0}

            pred, loss_value = sess.run(
                    [prediction,  loss,],
                    feed_dict=feed_dict)
        

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                sec_per_batch = float(duration)
                print '%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                     % (datetime.now(), step, loss_value,
                                FLAGS.batch_size/duration, sec_per_batch)

            predictions.extend(pred.tolist())
            labels.extend(batch_y.tolist())
            # print pred
            # print batch_y

        print labels
        print predictions
        acc = metrics.accuracy_score(labels, predictions)
        print 'acc:', acc*100



def main(argv):
    st = time.time() 
    print 'start loading data'
    dataset = Dataset(g_.IMAGE_LIST_TEST, subtract_mean=True, name='test')
    print 'done loading data, time=', time.time() - st

    test(dataset, FLAGS.weights)


if __name__ == '__main__':
    main(sys.argv)


