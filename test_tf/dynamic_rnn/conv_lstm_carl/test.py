import tensorflow as tf
import numpy as np
from cell import ConvLSTMCell
from cell import ConvGRUCell

batch_size = 2
timesteps = 6
height = 5
width = 4
shape = [height, width] # [height, width]
kernel = [3, 3]
channels = 3
filters = 12
# Create a placeholder for videos.
inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])

def t1():
    # Add the ConvLSTM step.
    cell = ConvLSTMCell(shape, filters, kernel)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
    with tf.Session() as sess:
        inp = np.random.normal(size=(timesteps, batch_size, height, width, channels))
        sess.run(tf.global_variables_initializer())
        output, cell_and_hidden_state = sess.run([outputs, state], feed_dict={inputs:inp})
        print("output shape:",output.shape) # output:[time, batch_size, height, width, width, num_filter]
        print("cell_and_hidden_state:", cell_and_hidden_state)

def t2():
    # There's also a ConvGRUCell that is more memory efficient.
    cell = ConvGRUCell(shape, filters, kernel)
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)
    with tf.Session() as sess:
        inp = np.random.normal(size=(timesteps, batch_size, height, width, channels))
        sess.run(tf.global_variables_initializer())
        output, cell_and_hidden_state = sess.run([outputs, state], feed_dict={inputs:inp})
        print("output shape:",output.shape) # output:[time, batch_size, height, width, width, num_filter]
        print("cell_and_hidden_state:", cell_and_hidden_state)

def t3():
    # It's also possible to enter 2D input or 4D input instead of 3D.
    shape = [100]
    kernel = [3]
    inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
    cell = ConvLSTMCell(shape, filters, kernel)
    outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)
    with tf.Session() as sess:
        inp = np.random.normal(size=(timesteps, batch_size, height, width, channels))
        sess.run(tf.global_variables_initializer())
        output, cell_and_hidden_state = sess.run([outputs, state], feed_dict={inputs:inp})
        print("output shape:",output.shape) # output:[time, batch_size, height, width, width, num_filter]
        print("cell_and_hidden_state:", cell_and_hidden_state)

def t4():
    shape = [50, 50, 50]
    kernel = [1, 3, 5]
    inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
    cell = ConvGRUCell(shape, filters, kernel)
    outputs, state= tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)
    with tf.Session() as sess:
        inp = np.random.normal(size=(timesteps, batch_size, height, width, channels))
        sess.run(tf.global_variables_initializer())
        output, cell_and_hidden_state = sess.run([outputs, state], feed_dict={inputs:inp})
        print("output shape:",output.shape) # output:[time, batch_size, height, width, width, num_filter]
        print("cell_and_hidden_state:", cell_and_hidden_state)


t1()
#t2()
#t3()
#t4()
