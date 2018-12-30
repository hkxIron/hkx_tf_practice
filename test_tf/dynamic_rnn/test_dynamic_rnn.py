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
X = np.random.randn(batch_size, timestep_size, input_dim) # X:[batch,timestep,input_dim]
# 第二个example长度为6
X[1,6:] = 0
X_lengths = [timestep_size, 6]
print("X:",X)
#sys.exit(-1)
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

output_all_hidden_states, last_cell_and_hidden_state = tf.nn.dynamic_rnn(  # 准确的讲是: all_hidden_state, last_cell_and_hidden_state
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n({"output_all_hidden_states": output_all_hidden_states,
                                 "last_cell_and_hidden_state": last_cell_and_hidden_state},
                                n=1,
                                feed_dict=None)

print("result: ",result) # state:(cell_state=[batch,hidden],hidden_state=[batch,hidden])
assert(result[0]["output_all_hidden_states"].shape == (batch_size, timestep_size, hidden_size))
# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert((result[0]["output_all_hidden_states"][1,7,:] == np.zeros(cell.output_size)).all())