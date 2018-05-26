# -*- coding:utf-8 -*-
# 本文来源于blog:http://blog.csdn.net/jerr__y/article/details/61195257
# github:https://github.com/yongyehuang/Tensorflow-Tutorial/blob/master/Tutorial_5%20-%20An%20understandable%20example%20to%20implement%20Multi-LSTM%20for%20MNIST.ipynb
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

# 设置 GPU 按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# 首先导入数据，看一下数据的形式
mnist = input_data.read_data_sets(train_dir='D:\\public_code\\hkx_tf_practice\\data\\mnist-tf\\', one_hot=True)

# 一、设置好模型用到的各个超参数
lr = 1e-3
input_size = 28      # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
timestep_size = 28   # 时序持续长度为28，即每做一次预测，需要先输入28行
hidden_size = 256    # 隐含层的数量
layer_num = 2        # LSTM layer 的层数
class_num = 10       # 最后输出分类类别数量，如果是回归预测的话应该是 1

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32, [])  # 注意类型必须为 tf.int32, batch_size = 128
keep_prob = tf.placeholder(tf.float32, [])



# 二、开始搭建 LSTM 模型，其实普通 RNNs 模型也一样
# 把784个点的字符信息还原成 28 * 28 的图片
# 下面几个步骤是实现 RNN / LSTM 的关键
####################################################################
# **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, 28, 28])

# # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
# lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

# # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
# lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

# # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
# mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
# mlstm_cell = rnn.MultiRNNCell([lstm_cell for _ in range(layer_num)] , state_is_tuple=True)

# 在 tf 1.0.0 版本中，可以使用上面的 三个步骤创建多层 lstm， 但是在 tf 1.2.1 版本中，可以通过下面方式来创建
def lstm_cell():
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple = True)

# **步骤5：用全零来初始化state
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

# **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
# ** 所以，可以取 hidden_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size], 注意：此处的2是指 cell_state以及hidden_state
# ** 或者，可以取 hidden_state = state[-1][1] 作为最后输出,即最后一层的hidden_state
# ** 最后输出维度是 [batch_size, hidden_size]
# outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# h_state = state[-1][1]

# *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
# 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数，我们可以用它来展开实现LSTM按时间步迭代。
# **步骤6：方法二，按时间步展开计算
outputs = list()
state = init_state
with tf.variable_scope('RNN'):
    for timestep in range(timestep_size):
        if timestep > 0:
            tf.get_variable_scope().reuse_variables()
        # 这里的state保存了每一层 LSTM 的状态
        # X[:,timestep,:]: [batch,hidden] state:(cell_state=[batch,hidden],hidden_state=[batch,hidden]) cell_output:[batch,hidden], 其值为hidden_state
        (cell_output, state) = mlstm_cell(X[:, timestep, :],state) # 调用 __call__函数,此处的cell_output实际上就是每次的hidden_state,而此处的state 包含(cell_state,hidden_state)
        if timestep==0: print("state:",state, " cell_output:",cell_output)
        outputs.append(cell_output)
h_state = outputs[-1] # 最后一个timestep的 hidden_state, shape为:[batch,hidden]

# 三、最后设置 loss function 和 优化器，展开训练并完成测试
############################################################################
# 以下部分其实和之前写的多层 CNNs 来实现 MNIST 分类是一样的。
# 只是在测试的时候也要设置一样的 batch_size.

# 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
# 首先定义 softmax 的连接权重矩阵和偏置
# out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
# out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
# 开始训练和测试
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1,shape=[class_num]), dtype=tf.float32)
y_pred = tf.nn.softmax(tf.matmul(h_state, W) + bias)


# 损失和评估函数
cross_entropy = -tf.reduce_mean(y * tf.log(y_pred)) # y是one_hot向量
train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())
_batch_size = 128
for i in range(2000):
    batch = mnist.train.next_batch(_batch_size)
    if (i+1)%20 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={_X:batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("Iter %d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy))
    sess.run(train_op, feed_dict={_X: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

# 计算测试数据的准确率
print("test accuracy %g"% sess.run(accuracy, feed_dict={_X: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size:mnist.test.images.shape[0]}))


# 查看LSTM的工作原理
import matplotlib.pyplot as plt
print("example label: ",mnist.train.labels[4])
X3 = mnist.train.images[4]
img3 = X3.reshape([28, 28])
plt.imshow(img3, cmap='gray')
plt.show()
plt.title("3")

X3.shape = [-1, 784]
y_batch = mnist.train.labels[0]
y_batch.shape = [-1, class_num]
X3_outputs = np.array(sess.run(outputs, feed_dict={ _X: X3, y: y_batch, keep_prob: 1.0, batch_size: 1}))
print(X3_outputs.shape) # (28,1,256)，即[timestep_size,1,hidden_size]
X3_outputs.shape = [28, hidden_size]
print(X3_outputs.shape) # (28,256)，即[timestep_size,hidden_size]

# -------------------
h_W = sess.run(W, feed_dict={ _X:X3, y: y_batch, keep_prob: 1.0, batch_size: 1})
h_bias = sess.run(bias, feed_dict={ _X:X3, y: y_batch, keep_prob: 1.0, batch_size: 1})
h_bias.shape = [-1, 10]

bar_index = range(class_num)
for i in range(X3_outputs.shape[0]):
    plt.subplot(7, 4, i+1) # 4*7=28个图
    X3_h_shate = X3_outputs[i, :].reshape([-1, hidden_size])
    pro = sess.run(tf.nn.softmax(tf.matmul(X3_h_shate, h_W) + h_bias)) # 计算输出概率
    plt.bar(bar_index, pro[0], width=0.2 , align='center') # 画柱状图
    plt.axis('off')
plt.show()

