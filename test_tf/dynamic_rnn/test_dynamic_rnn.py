#coding=utf-8
#测试 dynamic_rnn
import tensorflow as tf
import numpy as np
import sys
# 创建输入数据
batch_size=2
hidden_size=3
timestep_size=10
input_dim=2
np.random.seed(0)
X = np.random.randn(batch_size, timestep_size, input_dim) # X:[batch,timestep,input_dim]
# 第二个example长度为6
X[1,6:] = 0
X_lengths = [timestep_size, 6]
print("X:",X)
#sys.exit(-1)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, dtype=tf.float64)

output_all_hidden_states, last_cell_and_hidden_state = tf.nn.dynamic_rnn(  # 准确的讲是: all_hidden_state, last_cell_and_hidden_state
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    initial_state=initial_state,
    inputs=X)

# 也可以这样写
state_h = last_cell_and_hidden_state.h
state_c = last_cell_and_hidden_state.c

result = tf.contrib.learn.run_n({"output_all_hidden_states": output_all_hidden_states,
                                 "last_cell_and_hidden_state": last_cell_and_hidden_state},
                                n=1,
                                feed_dict=None)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("state_h:", sess.run(state_h)) # (batch, hidden)
    print("state_c:", sess.run(state_c)) # (batch, hidden)


print("result: ", result) # output_all_hidden_states: (batch_size, timestep_size, hidden_size)
# last_cell_and_hidden_state: c: (batch_size, hidden_size) h:(batch_size, hidden_size)
assert(result[0]["output_all_hidden_states"].shape == (batch_size, timestep_size, hidden_size))
# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert((result[0]["output_all_hidden_states"][1,7,:] == np.zeros(cell.output_size)).all())

"""
result:  [{'output_all_hidden_states': array([[[-0.01792841,  0.06766472,  0.21896471],
        [-0.02830377,  0.05916942,  0.24380742],
        [-0.06613405,  0.03208565,  0.197649  ],
        [-0.16665272,  0.07200808,  0.30682386],
        [-0.04264998,  0.01821844,  0.33329152],
        [-0.23735811, -0.00636422,  0.14689062],
        [-0.11930104, -0.04945073,  0.13804055],
        [-0.07136332, -0.06964927,  0.04687655],
        [ 0.07956828, -0.11617447, -0.15837592],
        [-0.0722373 , -0.02494699, -0.10694959]],

       [[ 0.05130406, -0.0083656 , -0.02975716],
        [-0.04706222,  0.05071078,  0.08926167],
        [-0.03614264, -0.00392445, -0.02043467],
        [-0.23557497, -0.00541252, -0.01616381],
        [-0.11207901, -0.03487944, -0.03386102],
        [ 0.05865898, -0.09473747, -0.15177235],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]]), 
        'last_cell_and_hidden_state': LSTMStateTuple(c=array([[-0.12265607, -0.05301387, -0.23669567],
       [ 0.09114471, -0.20438412, -0.53840047]]), h=array([[-0.0722373 , -0.02494699, -0.10694959],
       [ 0.05865898, -0.09473747, -0.15177235]]))}]
"""