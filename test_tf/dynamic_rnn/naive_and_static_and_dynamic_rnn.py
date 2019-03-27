#原文：https://blog.csdn.net/Rt_hai/article/details/81911236

import tensorflow as tf
import numpy as np
import time

batch_size=3
num_units=2
num_steps=10
input_dim=4

np.random.seed(0)
input=np.random.randn(batch_size,num_steps,input_dim)
# 第一个样本为10,第二个长度为6,第3个为5
input[1,6:]=0
input[2,5:]=0

def navie_rnn():
    # 手动for循环实现rnn
    # [batch_size, num_steps, input_dim]
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,num_steps,input_dim],name='input_x')
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units)
    initial_state=lstm_cell.zero_state(batch_size,dtype=tf.float32) # batch_size
    outputs=[]
    # 用for循环实现 rnn
    with tf.variable_scope('RNN'):
        for i in range(num_steps):
            if i > 0 : # 后面的需要复用以前的变量
                # print(tf.get_variable_scope())
                tf.get_variable_scope().reuse_variables()
            output=lstm_cell(x[:,i,:], initial_state)
            outputs.append(output)

    with tf.Session() as sess:
        init_op=tf.initialize_all_variables()
        sess.run(init_op)
        np.set_printoptions(threshold=np.NAN)
        output_and_cell_hidden_state=sess.run(outputs, feed_dict={x:input})
        print("output len:", len(output_and_cell_hidden_state)) # timestep
        print("output_all_hidden_states:", output_and_cell_hidden_state)

"""
output_all_hidden_states: 
[(array([[-0.01644995,  0.14463949], # hidden:[time_step, hidden_num]
       [ 0.19558035,  0.0298133 ],
       [-0.25547215, -0.18766902]], dtype=float32), 
       LSTMStateTuple(c=array([[-0.08766414,  0.5195793 ], # cell_state:[time_step, hidden_num]
       [ 0.38460055,  0.08481689],
       [-0.36924696, -0.33556435]], dtype=float32), 
       h=array([[-0.01644995,  0.14463949], # hidden_state:[time_step, hidden_num]
       [ 0.19558035,  0.0298133 ],
       [-0.25547215, -0.18766902]], dtype=float32))), 

 (array([[ 0.16158722,  0.03876749],
       [ 0.03049494,  0.06697942],
       [-0.15117253, -0.23054948]], dtype=float32), 
       LSTMStateTuple(c=array([[ 0.45025793,  0.07745986],
       [ 0.1419479 ,  0.34481406],
       [-0.23012838, -0.36518988]], dtype=float32), 
       h=array([[ 0.16158722,  0.03876749],
       [ 0.03049494,  0.06697942],
       [-0.15117253, -0.23054948]], dtype=float32))), 
(array([[-0.13701959,  0.18494599],
       [-0.21526287, -0.35160366],
       [ 0.01329887,  0.02338566]], dtype=float32), 
       LSTMStateTuple(c=array([[-0.36419722,  0.44705847],
       [-0.2744808 , -0.5217313 ],
       [ 0.02416002,  0.06248398]], dtype=float32), 
       h=array([[-0.13701959,  0.18494599],
       [-0.21526287, -0.35160366],
       [ 0.01329887,  0.02338566]], dtype=float32))), 
(array([[ 0.04295933,  0.11074147],
       [ 0.03908001,  0.05295254],
       [ 0.05995375, -0.18583655]], dtype=float32), 
       LSTMStateTuple(c=array([[ 0.11707453,  0.25508565],
       [ 0.11864971,  0.26337543],
       [ 0.08220273, -0.3437422 ]], dtype=float32), 
       h=array([[ 0.04295933,  0.11074147],
       [ 0.03908001,  0.05295254],
       [ 0.05995375, -0.18583655]], dtype=float32))), 
(array([[ 0.11562704,  0.02488268],
       [-0.45728686,  0.00868754],
       [-0.09092047,  0.03746721]], dtype=float32), 
       LSTMStateTuple(c=array([[ 0.28796098,  0.06206108],
       [-0.61087465,  0.02019263],
       [-0.12676495,  0.14790235]], dtype=float32), 
       h=array([[ 0.11562704,  0.02488268],
       [-0.45728686,  0.00868754],
       [-0.09092047,  0.03746721]], dtype=float32))), 
(array([[-0.13612318, -0.5462203 ],
       [-0.20313029,  0.02875011],
       [-0.08026256, -0.24774924]], dtype=float32), LSTMStateTuple(c=array([[-0.28060728, -0.74994177],
       [-0.2923849 ,  0.07557692],
       [-0.17814335, -0.36035088]], dtype=float32), h=array([[-0.13612318, -0.5462203 ],
       [-0.20313029,  0.02875011],
       [-0.08026256, -0.24774924]], dtype=float32)))]
...
        """



"""
如果觉得写for循环比较麻烦，则可以使用tf.nn.static_rnn函数，这个函数就是使用for循环实现的LSTM 
"""

def static_rnn():
    print("="*100)
    print("static_rnn")
    # [batch_size, num_steps, input_dim]
    x=tf.placeholder(dtype=tf.float32,shape=[batch_size,num_steps,input_dim],name='input_x')
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units)
    initial_state=lstm_cell.zero_state(batch_size,dtype=tf.float32) # batch_size
    # x:[batch_size, num_steps,input_dim],type:placeholder
    # input_tensor_list:num_steps*[batch_size,input_dim],type:list,即 num_steps个[batch_size, input_dim]
    input_tensor_list = tf.unstack(x, axis=1)
    output, state = tf.nn.static_rnn(cell=lstm_cell,
                                     inputs=input_tensor_list,
                                     sequence_length=[10, 6, 5], # 每个序列的长度
                                     initial_state=initial_state)
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        np.set_printoptions(threshold=np.NAN)
        output_all_hidden_states, last_cell_and_hidden_state = sess.run([output, state], feed_dict={x: input})
        #result1 = np.asarray(result1)
        #result2 = np.asarray(result2)
        print("output_hidden:", output_all_hidden_states)
        print("last_cell_and_hidden:", last_cell_and_hidden_state)


"""
不论dynamic_rnn还是static_rnn，每个batch的序列长度都是一样的（不足的话自己要去padding），不同的是dynamic会根据 sequence_length 中止计算。另外一个不同是dynamic_rnn动态生成graph 
但是dynamic_rnn不同的batch序列长度可以不一样，例如第一个batch长度为10，第二个batch长度为20，但是static_rnn不同的batch序列长度必须是相同的，都必须是num_steps 
下面使用dynamic_rnn来实现不同batch之间的序列长度不同
"""

def dynamic_rnn():
    print("="*100)
    print("dynamic_rnn")
    # [batch_size, num_steps, input_dim]
    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, input_dim], name='input')  # None 表示序列长度不定, batch与timestep长度均可`不固定
    lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(num_units, name="new_dynamic")
    initial_state=lstm_cell.zero_state(batch_size, dtype=tf.float32) # batch_size

    input = np.random.randn(batch_size, num_steps, input_dim)
    input2 = np.random.randn(batch_size, num_steps+1, input_dim) # 第2个input的timestep=num_step+1
    output_all_hidden_states, last_cell_and_hidden_state = tf.nn.dynamic_rnn(lstm_cell, inputs=x, initial_state=initial_state)

    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        np.set_printoptions(threshold=np.NAN)

        output_all_hidden_states_out, last_cell_and_hidden_state_out = \
            sess.run([output_all_hidden_states, last_cell_and_hidden_state],
                                     feed_dict={x: input})  # 序列长度为10 x:[batch_size,num_steps,input_dim],此时LSTM个数为10个，或者说循环10次LSTM
        print("======原始======")
        print("output_hidden:", output_all_hidden_states_out) # [batch, num_steps, hidden_num]
        print("last_cell_and_hidden:", last_cell_and_hidden_state_out)
        output_all_hidden_states_out, last_cell_and_hidden_state_out \
            = sess.run([output_all_hidden_states, last_cell_and_hidden_state],
                                     feed_dict={x: input2})  # 序列长度为20 x:[batch_size,num_steps+1,input_dim],此时LSTM个数为20个，或者说循环20次LSTM
        print("======长度加1======")
        print("output_hidden:", output_all_hidden_states_out)
        print("last_cell_and_hidden:", last_cell_and_hidden_state_out)
        #result1 = np.asarray(output_all_hidden_states_out)
        #result2 = np.asarray(result2)


"""
不论dynamic_rnn还是static_rnn，每个batch的序列长度都是一样的（不足的话自己要去padding），不同的是dynamic会根据 sequence_length 中止计算。另外一个不同是dynamic_rnn动态生成graph 
但是dynamic_rnn不同的batch序列长度可以不一样，例如第一个batch长度为10，第二个batch长度为20，但是static_rnn不同的batch序列长度必须是相同的，都必须是num_steps 

序列短的要比序列长的运行的快，dynamic_rnn比static_rnn快的原因是：
dynamic_rnn运行到序列长度后自动停止，不再运行，而static_rnn必须运行完num_steps才停止 

static_rnn seq_len:10		0.5824410915374756
static_rnn seq_len:100		1.1389522552490234
dynamic_rnn seq_len:10		0.31914639472961426
dynamic_rnn seq_len:100		0.7290496826171875
"""
def dynamic_vs_static_time_cost():
    print("="*100)
    print("dynamic_vs_static_time")

    num_step = 100
    input_dim = 8
    batch_size = 2
    num_unit = 64

    input_data = np.random.randn(batch_size, num_step, input_dim)
    x_dynamic = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_step, input_dim])
    seq_len = tf.placeholder(dtype=tf.int32, shape=[batch_size]) # 同一个batch中不同样本的序列长度
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_unit, name="time_cost")
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    output_dynamic, state_dynamic = tf.nn.dynamic_rnn(lstm_cell,
                                                      inputs=x_dynamic,
                                                      sequence_length=seq_len, # 每个序列的长度可能不一样
                                                      initial_state=initial_state)

    x_static = tf.unstack(x_dynamic, axis=1)
    output_static, state_static = tf.nn.static_rnn(lstm_cell,
                                                   inputs=x_static,
                                                   sequence_length=seq_len,
                                                   initial_state=initial_state)

    print('begin train...')
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        for i in range(100):
            sess.run([output_static, state_static],
                     feed_dict={x_dynamic: input_data,
                                seq_len: [10] * batch_size}) # batch中的序列长度均设为 10

        time1 = time.time()
        for i in range(100):
            sess.run([output_static, state_static],
                     feed_dict={x_dynamic: input_data,
                                seq_len: [10] * batch_size})
        time2 = time.time()
        print('static_rnn seq_len:10\t\t{}'.format(time2 - time1))

        for i in range(100):
            sess.run([output_static, state_static],
                     feed_dict={x_dynamic: input_data,
                                seq_len: [100] * batch_size}) # batch中的序列长度均设为 100
        time3 = time.time()
        print('static_rnn seq_len:100\t\t{}'.format(time3 - time2))

        for i in range(100):
            sess.run([output_dynamic, state_dynamic],
                     feed_dict={x_dynamic: input_data,
                                seq_len: [10] * batch_size})
        time4 = time.time()
        print('dynamic_rnn seq_len:10\t\t{}'.format(time4 - time3))

        for i in range(100):
            sess.run([output_dynamic, state_dynamic],
                     feed_dict={x_dynamic: input_data,
                                seq_len: [100] * batch_size})
        time5 = time.time()
        print('dynamic_rnn seq_len:100\t\t{}'.format(time5 - time4))

navie_rnn()
static_rnn()
dynamic_rnn()
dynamic_vs_static_time_cost()
