import numpy as np
import os,sys,inspect
import cv2 # need to import before tf, issue:https://github.com/tensorflow/tensorflow/issues/1541
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
tf.app.flags.DEFINE_string('n_views', 12, 
                            """Number of views rendered from a mesh.""")
tf.app.flags.DEFINE_string('caffemodel', '', 
                            """finetune with a model converted by caffe-tensorflow""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 20.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.05  # Learning rate decay factor.

np.set_printoptions(precision=3)



def train(dataset_train, dataset_val, ckptfile='', caffemodel=''):
    print 'train() called'
    is_finetune = bool(ckptfile)
    batch_size = FLAGS.batch_size

    data_size = dataset_train.size()
    print 'training size:', data_size

    with tf.Graph().as_default():
        startstep = 0 if not is_finetune else int(ckptfile.split('-')[-1])
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


        # must be after merge_all_summaries
        validation_loss = tf.placeholder('float32', shape=(), name='validation_loss')
        validation_summary = tf.scalar_summary('validation_loss', validation_loss)
        validation_acc = tf.placeholder('float32', shape=(), name='validation_accuracy')
        validation_acc_summary = tf.scalar_summary('validation_accuracy', validation_acc)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.initialize_all_variables()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
        
        if is_finetune:
            saver.restore(sess, ckptfile)
            print 'restore variables done'
        elif caffemodel:
            sess.run(init_op)
            model.load_alexnet(sess, caffemodel)
            print 'loaded pretrained caffemodel:', caffemodel
        else:
            # from scratch
            sess.run(init_op)
            print 'init_op done'

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
                                                graph=sess.graph) 

        step = startstep
        for epoch in xrange(100):
            print 'epoch:', epoch

            dataset_train.shuffle()
            # dataset_val.shuffle()

            for batch_x, batch_y in dataset_train.batches(batch_size):
                # print batch_x_v[0,0,:]
                # print batch_y

                if step >= FLAGS.max_steps:
                    break
                step += 1

                start_time = time.time()
                feed_dict = {image_: batch_x,
                             y_ : batch_y,
                             keep_prob_: 0.5,
                             phase_train_: True}

                _, loss_value, logitsyo, _ = sess.run(
                        [train_op, loss, logits, print_op],
                        feed_dict=feed_dict)

                # print batch_y
                # print logitsyo.max(), logitsyo.min()

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0 or step < 30:
                    sec_per_batch = float(duration)
                    print '%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                         % (datetime.now(), step, loss_value,
                                    FLAGS.batch_size/duration, sec_per_batch)

                        
                # val
                if step % 100 == 0:# and step > 0:
                    val_losses = []
                    
                    val_logits = []
                    predictions = np.array([])
                    val_y = []
                    for val_step, (val_batch_x, val_batch_y) in \
                            enumerate(dataset_val.sample_batches(batch_size, g_.VAL_SAMPLE_SIZE)):
                            # enumerate(dataset_val.batches(batch_size)):
                        val_feed_dict = {image_: val_batch_x, 
                                         y_  : val_batch_y,
                                         keep_prob_: 1.0,
                                         phase_train_: False }
                        val_loss, pred, val_logit ,_= sess.run([loss, prediction, logits, print_op], feed_dict=val_feed_dict)

                        val_losses.append(val_loss)
                        val_logits.extend(val_logit.tolist())
                        predictions = np.hstack((predictions, pred))
                        val_y.extend(val_batch_y)

                    val_logits = np.array(val_logits)
                    # print val_logits
                    # print val_y
                    # print predictions
                    # print val_logits[0].tolist()
                    
                    # val_logits.dump('val_logits.npy')
                    # predictions.dump('predictions.npy')
                    # np.array(val_y).dump('val_y.npy')

                    val_loss = np.mean(val_losses)
                    acc = metrics.accuracy_score(val_y[:predictions.size], np.array(predictions))
                    print '%s: step %d, validation loss=%.4f, acc=%f' %\
                            (datetime.now(), step, val_loss, acc*100.)

                    # validation summary
                    val_loss_summ = sess.run(validation_summary,
                            feed_dict={validation_loss: val_loss})
                    val_acc_summ = sess.run(validation_acc_summary, 
                            feed_dict={validation_acc: acc})
                    summary_writer.add_summary(val_loss_summ, step)
                    summary_writer.add_summary(val_acc_summ, step)
                    summary_writer.flush()


                if step % 100 == 0:
                    # print 'running fucking summary'
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                if step % 200  == 0 or (step+1) == FLAGS.max_steps \
                        and step > startstep:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)



def main(argv):
    st = time.time() 
    print 'start loading data'
    dataset_train = Dataset(g_.IMAGE_LIST_TRAIN, subtract_mean=True, name='train')
    dataset_val = Dataset(g_.IMAGE_LIST_VAL, subtract_mean=True, name='val')
    print 'done loading data, time=', time.time() - st

    train(dataset_train, dataset_val, FLAGS.weights, FLAGS.caffemodel)


if __name__ == '__main__':
    main(sys.argv)

