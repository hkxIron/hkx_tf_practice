#blog：https://blog.csdn.net/qq_35203425/article/details/81332807
import tensorflow as tf
import numpy as np

batch=3
time_step = 10
word_embedding_size=8

X = np.random.randn(batch, time_step, word_embedding_size)
# The second example is of length 6
# 第1个样本长度为10
X[1, 6:] = 0 # 第二个样本长度为6
X[2, 3:] = 0 # 第三个样本长度为3
X_lengths = [10, 6, 3] # 每个样本的时间长度
X_tensor = tf.convert_to_tensor(X)

# cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
# cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)


def get_lstm_cell(rnn_size):
    lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    return lstm_cell


# 多层lstm
num_layers=3
cell_fw = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size=2) for _ in range(num_layers)])
cell_bw = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size=2) for _ in range(num_layers)])

# output_fw: [batch, input_sequence_length, num_units],它的值为hidden_state
# output_bw: [batch, input_sequence_length, num_units],它的值为hidden_state
# (cell_state_fw, hidden_state_fw) = states_fw
# cell_state: [batch, num_units]
# hidden_state: [batch, num_units]
# states_fw: (LSTMTuple(cell_state, hidden_state),LSTMTuple(cell_state, hidden_state), ... ),
# rnn有n层,state_fw 的tuple的长度就会为n

(output_fw, output_bw), (states_fw, states_bw)= tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell_fw,
    cell_bw=cell_bw,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X_tensor # 将numpy 数组转为tensor
)


# tf中的所谓output一般指hidden_state,而非cell_state, cell_state中一般包含所有时记刻的cell_state,hidden_state
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('states forward\n', sess.run(states_fw))
    print('output forward\n', sess.run(output_fw))

"""
states forward:(同于有3层rnn堆叠,所以有3个LstmStateTuple,每个cell_state,hidden_state:[batch, hidden])
 (LSTMStateTuple(c=array([[0.08934514, 0.0252916 ],
       [0.10183962, 0.08388254],
       [0.01710488, 0.15693404]]), 
       h=array([[0.04100799, 0.01124952],
       [0.04779271, 0.04607952],
       [0.00859684, 0.08097949]])), 
  LSTMStateTuple(c=array([[0.008261  , 0.00685363],
       [0.00633035, 0.00531445],
       [0.00444411, 0.00240098]]), 
       h=array([[0.00413328, 0.00342911],
       [0.003164  , 0.00265853],
       [0.00221702, 0.00119927]])), 
  LSTMStateTuple(c=array([[0.00069844, 0.00072609],
       [0.00053869, 0.00055306],
       [0.00012883, 0.00013095]]), 
       h=array([[3.49215444e-04, 3.63058186e-04],
       [2.69345756e-04, 2.76540229e-04],
       [6.44161555e-05, 6.54760217e-05]])))
       
output forward: [batch, sequence_length, hidden],并不会对每层的rnn输出都有记录,只记录最后一层的输出
 [[[-6.53570424e-05 -5.69010553e-05]
  [-1.48919063e-04 -1.30709173e-04]
  [-1.34982635e-04 -1.17489866e-04]
  [-8.42723443e-05 -7.18594745e-05]
  [ 2.46086472e-05  2.94491369e-05]
  [ 1.58980690e-04  1.58226471e-04]
  [ 1.85471284e-04  1.93644684e-04]
  [ 2.39841969e-04  2.51280025e-04]
  [ 3.07909951e-04  3.19680885e-04]
  [ 3.49215444e-04  3.63058186e-04]]

 [[ 5.90403616e-05  5.40542858e-05]
  [ 1.71838657e-04  1.59065417e-04]
  [ 2.23293457e-04  2.12425061e-04]
  [ 2.49055450e-04  2.43348284e-04]
  [ 2.37535976e-04  2.42288489e-04]
  [ 2.69345756e-04  2.76540229e-04]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]]

 [[-5.85575409e-06 -2.88851583e-06]
  [ 6.37048683e-06  1.02783119e-05]
  [ 6.44161555e-05  6.54760217e-05]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]
  [ 0.00000000e+00  0.00000000e+00]]]
"""

