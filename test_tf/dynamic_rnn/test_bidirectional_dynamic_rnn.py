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

cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)

outputs, states = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell,
    cell_bw=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X_tensor # 将numpy 数组转为tensor
)

output_fw, output_bw = outputs
states_fw, states_bw = states

# tf中的所谓output一般指hidden_state,而非cell_state, cell_state中一般包含所有时记刻的cell_state,hidden_state
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    states_shape = tf.shape(states)
    print(states_shape.eval())
    c, h = states_fw
    o = output_fw
    print('c(last cell state) forward\n', sess.run(c))
    print('h(last hidden state) forward\n', sess.run(h))
    print('o(all hidden state seq) forward:\n', sess.run(o))

"""
c(last cell state) forward
 [[-0.12217524  0.28533151 -0.36844351 -0.43810071  0.00811558]
 [ 0.07820437 -0.92231635 -0.26364966  0.31294861  0.31548287]
 [-0.65065489 -0.32258168 -0.30470056  0.13921824  0.38474448]]
h(last hidden state) forward
 [[-0.04144077  0.12940451 -0.16907926 -0.21485672  0.00483757]
 [ 0.06429532 -0.56275855 -0.21832985  0.1266499   0.0818067 ]
 [-0.20172505 -0.12580037 -0.14340393  0.04755116  0.23248379]]
o(all hidden state seq) forward:
 [[[-0.04676045  0.02525628  0.15680289 -0.08090868 -0.00158162]
  [-0.13279093  0.07675365  0.10376204  0.02689012  0.17435146]
  [-0.18651114  0.0663562   0.13765341  0.07514173  0.23420578]
  [-0.21622866  0.10076028  0.1125673   0.0395987   0.15144544]
  [-0.37538322  0.22057878  0.07480336 -0.2497213  -0.04843769]
  [-0.14348758  0.23664133 -0.04841355 -0.12304318 -0.06879915]
  [-0.04373433  0.09694173  0.04338431 -0.24696993 -0.04512253]
  [-0.0200637   0.01545513  0.18276741 -0.15706653 -0.07947899]
  [ 0.12769996  0.13402019 -0.09276391 -0.19386652 -0.12720117]
  [-0.04144077  0.12940451 -0.16907926 -0.21485672  0.00483757]]

 [[-0.0282109  -0.01464982  0.06411071  0.06264488  0.06290473]
  [ 0.09929985 -0.02277209 -0.13806624  0.0066819   0.03737792]
  [ 0.11699781 -0.07623118  0.00165157  0.09244142  0.13118021]
  [-0.10898187 -0.07082201  0.0280239   0.1767859   0.16226114]
  [ 0.01326696 -0.2865498   0.00600045  0.31084136  0.44617576]
  [ 0.06429532 -0.56275855 -0.21832985  0.1266499   0.0818067 ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]]

 [[-0.20438708 -0.07382884  0.14202276  0.07297622  0.13386155]
  [-0.1072367  -0.34855129 -0.02832463  0.13459999  0.18509057]
  [-0.20172505 -0.12580037 -0.14340393  0.04755116  0.23248379]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]
  [ 0.          0.          0.          0.          0.        ]]]
"""