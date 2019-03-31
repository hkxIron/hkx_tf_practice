import tensorflow as tf
import numpy as np

# https://blog.csdn.net/sinat_26917383/article/details/71817742
# 卷积lstm
class BasicConvLSTMCell(tf.contrib.rnn.RNNCell):
  def __init__(self, shape, num_filters, kernel_size, forget_bias=1.0,
               input_size=None, state_is_tuple=True, activation=tf.nn.tanh, reuse=None):
    self._shape = shape # [height, width]
    self._num_filters = num_filters # 6
    self._kernel_size = kernel_size # [filter_height, filter_weight]
    self._size = tf.TensorShape(shape+[self._num_filters]) # [height, width, num_filter]

    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._reuse = reuse

  @property
  def state_size(self):
    return (tf.contrib.rnn.LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._size

  def __call__(self, inputs, state, scope=None):
    # we suppose inputs to be [time, batch_size, height, width, channel]
    # 注意: 由于是time_major为true,因此,每次输入的数据均为
    # inputs: [batch_size, height, width, channel]
    print("inputs shape:", inputs.get_shape) # [batch, height, width, channel]
    with tf.variable_scope(scope or "basic_convlstm_cell", reuse=self._reuse):
      if self._state_is_tuple:
        c, h = state
        print("c shape:", c.get_shape) # (4,3,3,6), h:[batch, height, width, num_filter]
        print("h shape:", h.get_shape)
      else:
        c, h = tf.split(value=state, num_or_size_splits=2, axis=3)

      # inp_channel: channel+num_filter
      inp_channel = inputs.get_shape().as_list()[-1]+self._num_filters
      print("inp_channel:", inp_channel)
      # out_channel:4*num_filter
      out_channel = self._num_filters * 4 # i, j, f, o
      # h:[batch, height, width, num_filter]
      # concat: [batch_size, height, width, channel]
      #  =>: [batch_size, height, width, channel+num_filter]
      concat = tf.concat([inputs, h], axis=3) # x_t, h_(t-1)
      # self._kernel_size: [filter_height, filter_width]
      # kernel: [filter_height, filter_width, inp_channel=channel+num_filter, out_channel]
      kernel = tf.get_variable('kernel', shape=self._kernel_size + [inp_channel, out_channel])
      # concat: [batch_size, height, width, width, channel+num_filter]
      #  => [batch_size, height, width, width, out_channel]
      concat = tf.nn.conv2d(concat, filter=kernel, strides=(1,1,1,1), padding='SAME')

      #  concat:[batch_size, height, width, width, out_channel=4*num_filter]
      # i:[batch_size, height, width, width, num_filter]
      i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

      # new_c:[batch_size, height, width, width, num_filter]
      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)
      if self._state_is_tuple:
        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat([new_c, new_h], 3)
      return new_h, new_state

if __name__ == '__main__':
  (time_steps, batch, height, width, channel) = [5, 4, 3, 3, 2]

  inputs=tf.placeholder(tf.float32, shape=[time_steps, batch, height, width, channel])
  cell = BasicConvLSTMCell(shape=[3, 3], num_filters=6, kernel_size=[3,3])
  outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype, time_major=True)
  with tf.Session() as sess:
    inp = np.random.normal(size=(time_steps, batch, height, width, channel))
    sess.run(tf.global_variables_initializer())
    output, cell_and_hidden_state = sess.run([outputs, state], feed_dict={inputs:inp})
    print("output shape:",output.shape) #(5,2,3,3,6), output:[time, batch_size, height, width, width, num_filter]
    print("cell_and_hidden_state:", cell_and_hidden_state)
