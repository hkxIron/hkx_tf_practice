# https://blog.csdn.net/uestc_c2_403/article/details/73368557
import tensorflow as tf
import numpy as np

import tensorflow as tf;
import numpy as np;

X = tf.random_normal(shape=[3, 5, 6], dtype=tf.float32)
X[1, 2:] = 0
X[2, 3:] = 0
X = tf.reshape(X, [-1, 5, 6])
X_lengths=[5,2,3]
#cell = tf.nn.rnn_cell.GRUCell(10)
fw_cell = tf.nn.rnn_cell.GRUCell(num_units=5, dtype=tf.float32)
bw_cell = tf.nn.rnn_cell.GRUCell(num_units=5, dtype=tf.float32)
#init_state = cell.zero_state(3, dtype=tf.float32)
#output, state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)
output, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=X, sequence_length=X_lengths,dtype=tf.float32, time_major=False)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(output))
    print(sess.run(state))

