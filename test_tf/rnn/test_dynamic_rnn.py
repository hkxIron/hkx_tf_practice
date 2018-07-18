# blog: https://blog.csdn.net/qq_23142123/article/details/78486303
#coding=utf-8
#测试 dynamic_rnn
# 假设场景为翻译，有2个句子，第1个句子有10个单词,第2个句子有6个单词,每个单词的embedding维度为2
import tensorflow as tf
import numpy as np
import sys
# 创建输入数据
batch_size=2
hidden_size=3
timestep_size=10 # timestep取的是最大的序列长度
input_dim=2
np.random.seed(0)
X = np.random.randn(batch_size, timestep_size, input_dim) # X:[batch,timestep,input_dim]
# 第二个example长度为6
X[1,6:] = 0 # 将第2个句子的后4个单词的embedding向量全置为0
X_lengths = [timestep_size, 6] # 分别设置两个句子的长度为[10,6]
print("X:", X)
print("X_lengths:", X_lengths)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True) # 一般都设成tuple，这样比较容易区分

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n({"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)

print("list of result: ", result) # list of dict: [ {"outputs": array[...] , 'last_states': array[...]}]

print("result: ",result[0]) # state:(cell_state=[batch,hidden],hidden_state=[batch,hidden])
assert(result[0]["outputs"].shape == (batch_size, timestep_size, hidden_size))
# 第二个example中的outputs超过6步(7-10步)的值应该为0
assert((result[0]["outputs"][1,7,:] == np.zeros(cell.output_size)).all())

"""
result:  
{
'outputs': array([[
        [-0.26704391,  0.00372268, -0.06960527], # 第1个句子的第1个hidden_state
        [-0.33990261,  0.06774519, -0.24866373], # 第1个句子的第2个hidden_state
        [-0.39665081, -0.00360473, -0.14212302], # 第1个句子的第3个hidden_state
        [-0.32046912,  0.02009141, -0.18959314],
        [-0.18888559,  0.05826969, -0.23720432],
        [-0.21813354,  0.09749746, -0.3367005 ],
        [-0.23823033,  0.05960983, -0.28300906],
        [-0.20850051,  0.06488494, -0.30551283],
        [-0.31413775,  0.03520423, -0.26472816],
        [-0.11911752,  0.00269063, -0.18399638]], # 这一行为第1个句子的最新的hidden_state 

       [[ 0.04911522,  0.05381059,  0.0451024 ],
        [ 0.00914185, -0.01320099,  0.07243182],
        [-0.21775088, -0.08769896,  0.09174084],
        [-0.07829152, -0.05035589,  0.10335326],
        [-0.33122035, -0.00233101, -0.04098983],
        [-0.20537287,  0.04739766, -0.11714717], # 这一行为第2个句子的最新的hidden_state
        [ 0.        ,  0.        ,  0.        ], # 由于句子长度只有6,因此7-10的hidden_state都是0
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]]),
         
'last_states': LSTMStateTuple(
c=array([[-0.25459499,  0.00674475, -0.42207152], # lstm中的cell_state
       [-0.45962667,  0.08796889, -0.21598014]]), 
h=array([[-0.11911752,  0.00269063, -0.18399638], # lstm中每个序列最后的hidden_state
       [-0.20537287,  0.04739766, -0.11714717]]))
}

从上面可以看出：
1. outputs实际上将每个序列的所有历史hidden_state都存储起来
2. last_states包含cell_state,hidden_state，都只有最新的状态
"""