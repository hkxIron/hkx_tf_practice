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

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n({"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

print("result: ",result[0]) # state:(cell_state=[batch,hidden],hidden_state=[batch,hidden])
assert(result[0]["outputs"].shape == (batch_size, timestep_size, hidden_size))
# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert((result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all())
