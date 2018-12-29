#coding=utf-8
import tensorflow as tf
import re
import numpy as np
import globals as g_

FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', g_.BATCH_SIZE,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_float('learning_rate', g_.INIT_LEARNING_RATE,
                            """Initial learning rate.""")

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
WEIGHT_DECAY_FACTOR = 0.001 # to make l2-regularizer about the same to c-e loss
DEFAULT_PADDING = 'SAME'

def _conv(name, phase_train, in_ ,ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING, batch_norm=False, group=1):
    
    n_kern = ksize[3]

    with tf.variable_scope(name, reuse=False) as scope:
        stddev = 1 / np.prod(ksize[:3], dtype=float) ** 0.5
        print name, 'stddev', stddev

        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, stddev=stddev, wd=0.0)
            conv = tf.nn.conv2d(in_, kernel, strides, padding=padding)
	else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, stddev=stddev, wd=0.0)
	    input_groups = tf.split(3, group, in_)
	    kernel_groups = tf.split(3, group, kernel)
            convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=padding)
	    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
	    # Concatenate the groups
	    conv = tf.concat(3, output_groups)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(conv, name=scope.name)

        if batch_norm:
            conv = batch_norm_layer(conv, phase_train, scope.name)

    print name, conv.get_shape().as_list()
    return conv


def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
            lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                    center=False, updates_collections=None, scope=scope+"_bn"),
            lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                    updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print name, pool.get_shape().as_list()
    return pool

def _fc(name, in_, outsize, dropout=1.0, activation='relu'):
    with tf.variable_scope(name, reuse=False) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        insize = in_.get_shape().as_list()[-1]
        stddev = (1 / float(insize)) ** 0.5
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize],
                                              stddev=stddev, wd=WEIGHT_DECAY_FACTOR)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        # fc = tf.nn.bias_add(tf.matmul(in_, weights), biases)
        fc = tf.matmul(in_, weights) + biases
        if activation == 'relu':
            fc = tf.nn.relu(fc, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)

    print name, fc.get_shape().as_list()
    return fc
    

def input():
    model = g_.MODEL
    if model.lower() == 'alexnet':
        w, h = 227, 227
    elif model.lower() == 'vgg16':
        w, h = 224, 224

    image_ = tf.placeholder('float32', shape=(None, h, w, 3), name='image')
    y_ = tf.placeholder('uint16', shape=[None], name='y')

    return image_, y_


def inference(image, keep_prob, phase_train):
    model = g_.MODEL
    print 'inferencing:', model
    if model.lower() == 'alexnet':
        return inference_alexnet(image, keep_prob, phase_train)
    elif model.lower() == 'vgg16':
        return inference_vgg16(image, keep_prob, phase_train)

def inference_alexnet(image, keep_prob, phase_train):
    """
    images: N x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    phase_train: bool, True for training
    """

    conv1 = _conv('conv1', phase_train, image, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', batch_norm=g_.BATCH_NORM)
    conv1 = tf.nn.local_response_normalization(conv1, depth_radius=2,
                                                  alpha=2e-5,
                                                  beta=0.75,
                                                  bias=1.0,
                                                  name='lrn1')
    pool1 = _maxpool('pool1', conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = _conv('conv2', phase_train, pool1, [5, 5, 96, 256], batch_norm=g_.BATCH_NORM, group=2)
    conv2 = tf.nn.local_response_normalization(conv2, depth_radius=2,
                                                  alpha=2e-5,
                                                  beta=0.75,
                                                  bias=1.0,
                                                  name='lrn2')
    pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv3 = _conv('conv3', phase_train, pool2, [3, 3, 256, 384], batch_norm=g_.BATCH_NORM)
    conv4 = _conv('conv4', phase_train, conv3, [3, 3, 384, 384], batch_norm=g_.BATCH_NORM, group=2)
    conv5 = _conv('conv5', phase_train, conv4, [3, 3, 384, 256], batch_norm=g_.BATCH_NORM, group=2)
    pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        
    pool5 = _flatten_featuremap(pool5)

    fc6 = _fc('fc6', pool5, 4096, dropout=keep_prob)
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob)
    fc8 = _fc('fc8', fc7, g_.N_CLASSES, activation=None)

    return fc8
    

def inference_vgg16(image, keep_prob, phase_train):
    """
    images: N x W x H x C tensor
    keep_prob: keep rate for dropout in fc layers
    phase_train: bool, True for training
    """
    conv1_1 = _conv('conv1_1', phase_train, image, [3, 3, 3, 64], [1, 1, 1, 1], batch_norm=g_.BATCH_NORM)
    conv1_2 = _conv('conv1_2', phase_train, conv1_1, [3, 3, 64, 64], [1, 1, 1, 1], batch_norm=g_.BATCH_NORM)
    pool1 = _maxpool('pool1', conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    conv2_1 = _conv('conv2_1', phase_train, pool1, [3, 3, 64, 128], batch_norm=g_.BATCH_NORM)
    conv2_2 = _conv('conv2_2', phase_train, conv2_1, [3, 3, 128, 128], batch_norm=g_.BATCH_NORM)
    pool2 = _maxpool('pool2', conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    conv3_1 = _conv('conv3_1', phase_train, pool2, [3, 3, 128, 256], batch_norm=g_.BATCH_NORM)
    conv3_2 = _conv('conv3_2', phase_train, conv3_1, [3, 3, 256, 256], batch_norm=g_.BATCH_NORM)
    conv3_3 = _conv('conv3_3', phase_train, conv3_2, [3, 3, 256, 256], batch_norm=g_.BATCH_NORM)
    pool3 = _maxpool('pool3', conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    conv4_1 = _conv('conv4_1', phase_train, pool3, [3, 3, 256, 512], batch_norm=g_.BATCH_NORM)
    conv4_2 = _conv('conv4_2', phase_train, conv4_1, [3, 3, 512, 512], batch_norm=g_.BATCH_NORM)
    conv4_3 = _conv('conv4_3', phase_train, conv4_2, [3, 3, 512, 512], batch_norm=g_.BATCH_NORM)
    pool4 = _maxpool('pool4', conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    conv5_1 = _conv('conv5_1', phase_train, pool4, [3, 3, 512, 512], batch_norm=g_.BATCH_NORM)
    conv5_2 = _conv('conv5_2', phase_train, conv5_1, [3, 3, 512, 512], batch_norm=g_.BATCH_NORM)
    conv5_3 = _conv('conv5_3', phase_train, conv5_2, [3, 3, 512, 512], batch_norm=g_.BATCH_NORM)
    pool5 = _maxpool('pool5', conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        
    pool5 = _flatten_featuremap(pool5)

    fc6 = _fc('fc6', pool5, 4096, dropout=keep_prob)
    fc7 = _fc('fc7', fc6, 4096, dropout=keep_prob)
    fc8 = _fc('fc8', fc7, g_.N_CLASSES, activation=None)

    return fc8

def _flatten_featuremap(in_):
    dim = np.prod(in_.get_shape().as_list()[1:])
    return tf.reshape(in_, [-1, dim])

def load_model(sess, caffetf_modelpath, fc8=False):
    model = g_.MODEL
    print 'loading model:', model
    if model.lower() == 'alexnet':
        load_alexnet(sess, caffetf_modelpath, fc8)
    elif model.lower() == 'vgg16':
        load_vgg16(sess, caffetf_modelpath, fc8)

def load_alexnet(sess, caffetf_modelpath, fc8=False):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    print data_dict.keys()
    
    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = l
        _load_param(sess, name, data_dict[l], old_format=True)

    if fc8:
        _load_param(sess, 'fc8', data_dict['fc8'], old_format=True)

def load_vgg16(sess, caffetf_modelpath, fc8=False):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()
    layers = data_dict.keys()
    for l in layers:
        if l == 'fc8' and not fc8:
            continue
        _load_param(sess, l, data_dict[l])


def _load_param(sess, name, layer_data, old_format=False):
    if not old_format:
        w, b = layer_data['weights'], layer_data['biases']
    else:
        w, b = layer_data

    with tf.variable_scope(name, reuse=True):
        for subkey, data in zip(('weights', 'biases'), (w, b)):
            print 'loading ', name, subkey

            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))
            except ValueError as e: 
                print 'varirable not found in graph:', subkey, str(e)


def loss(logits, labels):
    
    labels = tf.cast(labels, tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits, labels, name='cross_entropy_per_example')
    loss_mean = tf.reduce_mean(ce, name='cross_entropy')

    # p = tf.Print(loss_mean, [loss_mean], 'cross entropy:')
    p = tf.no_op()
    tf.add_to_collection('losses', loss_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss'), p

def classify(fc8):
    softmax = tf.nn.softmax(fc8)
    y = tf.argmax(softmax, 1)
    return y

def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    print 'losses:', losses
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op
    


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
                           # tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def train(total_loss, global_step, data_size):
    num_batches_per_epoch = data_size / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        # opt = tf.train.AdamOptimizer(lr)
        # opt = tf.train.MomentumOptimizer(lr, 0.9)
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        print 'All grads:', grads

        # only fc6-8 weights and bias
        # grads = grads[-12:]
        # print 'fc8 grads', grads
    
    # apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad,var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
