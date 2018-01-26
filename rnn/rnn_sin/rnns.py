import tensorflow as tf
import numpy as np

batch_size=100
hidden_size = 50
num_steps = 20 # sequence的长度
input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
targets = tf.placeholder(tf.int32, [batch_size, num_steps])
current_input = np.asarray([[1.0,2.0],[0.3,0.6]])
use_dropout= True

# 单层lstm
def single_lstm():
    # 定义一个lstm结构,lstm中使用的变量也会在该函数中自动被声明。
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    if use_dropout:
        # 定义使用LSTM结构及训练时使用dropout,即在原始lstm上再加一层。
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.5)
    # 生成全0的初始化状态。和其他神经网络类似，在优化循环神经网络时，每次也会使用一个batch的训练样本，
    # 在以下代码中，batch_size给出了一个batch的大小。
    state = lstm_cell.zero_state(batch_size,tf.float32)
    # 定义损失函数
    loss =0.0
    # 虽然理论上循环神经网络可以处理任意长度的序列，但是在训练时为了避免梯度消散的问题，会规定一个最大的序列长度。
    # 在以下代码中，用num_steps来表示这个长度。
    for i in range(num_steps):
        # 在第一个时刻声明lstm结构中使用的变量，在之后的时刻都需要复用之前定义好的变量。
        if i>0:tf.get_variable_scope().reuse_variables()
        # 每一步处理时间序列中的一个时刻。将当前输入(current_input)和前一时刻状态
        # （state）传入定义的lstm结构可以得到当前lstm结构的输出lstm_output和更新后的状态state.
        lstm_output,state = lstm_cell(current_input,state)
        # 将当前时刻LSTM结构的输出传入一个全连接层得到最后的输出
        final_output = tf.contrib.layers.fully_connected(lstm_output)
        # 计算当前时刻输出的损失
        loss += calculate_loss(final_output,expected_output)

def stacked_lstm():
    # 多层循环神经网络
    # 定义一个lstm结构作为循环体的基础结构，也可以使用其他循环体结构
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    # 通过MulitRNNCell类实现深层循环神经网络中每一个时刻的前向传播过程。
    # 其中number_of_layers表示了有多少层，也就是从xt到ht需要经过多少个lstm结构
    number_of_layers = 5
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*number_of_layers)
    # 与原始的循环神经网络一样，可以通过zero_state函数来获取初始状态。
    state = stacked_lstm.zero_state(batch_size,tf.float32)
    # 计算前向传播
    for i in range(len(num_steps)):
        if i >0: tf.get_variable_scope().reuse_variables()
        stacked_lstm_output,state = stacked_lstm(current_input,state)
        final_output = tf.contrib.layers.fully_connected(stacked_lstm_output)
        loss += calculate_loss(final_output, expected_output)

