# Import useful packages
# TODO:此代码有问题
from keras import utils
import numpy as np
import tensorflow as tf
#%matplotlib inline
import matplotlib.pyplot as plt
# Reset the default graph for rerun notebook
#tf.reset_default_graph()
# Reset the random seed for reproducibility
np.random.seed(42)
tf.set_random_seed(42)

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/")
import tensorflow as tf
mnist = tf.keras.datasets.mnist # 包含了很多数据集，第一次使用需要下载
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape) # out: (60000, 28, 28)
print(y_train.shape) # out: (60000,)

n_samples = 5
"""
plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index+ 1)
    sample_image = X_train[index].reshape(28, 28)
    plt.imshow(sample_image, cmap="binary")
    plt.axis("off")
    plt.show()
"""

X= tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")  # [batch, width, height, channel]
y= tf.placeholder(shape=[None], dtype=tf.int64, name="y")

conv1_params = {
"filters": 256,
"kernel_size": 9,
"strides": 1,
"padding": "valid",
"activation": tf.nn.relu,
}
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params) # [batch, width=(28+0-9)/1+1=20, height, out_channel=256]

# Capsules
caps1_n_maps = 32
caps1_n_dims = 8 # 每个capsule有8维,在此这里就是以前cnn的8倍的filter
conv2_params = {
"filters": caps1_n_maps * caps1_n_dims, # 32*8, 但注意,这些filter之间并没有共享参数
"kernel_size": 9,
"strides": 2,
"padding": "valid",
"activation": tf.nn.relu
}
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params) # [batch, width=(20+0-9)//2+1=6,height=6, out_channel=32*8]

caps1_n_caps = 6*6*caps1_n_maps # 6*6*32 = 1152
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw") # [batch, 6*6*32, 8]
print("caps1_raw:{}".format(caps1_raw.shape))

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon) # s/|s|
        squash_factor = squared_norm / (1. + squared_norm) # |x|^2/(1+|x|^2)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


caps1_output = squash(caps1_raw, name="caps1_output") # [batch, caps1_n_caps=6*6*32, 8]

# 高层有10个语义capsule
caps2_n_caps = 10
caps2_n_dims = 16
init_sigma = 0.01

W_init = tf.random_normal(shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims), #[1, in_n1, out_n2, n2_dim, n1_dim]
                          stddev=init_sigma,
                          dtype=tf.float32,
                          name="W_init")

W = tf.Variable(W_init, name="W")
batch_size = tf.shape(X)[0]
print("batch size:",batch_size, " x shape:", X.shape)
# W好像不应该复制吧
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled") # [batch, in_n1, out_n2, n2_dim, n1_dim]

#----------
caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded") # [batch, caps1_n_caps=6*6*32, 8, 1]
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile") # [batch, caps1_n_caps=6*6*32, 1, 8, 1]
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled") # [batch, caps1_n_caps=6*6*32, 10, 8, 1]

# w_tiled:[batch, in_n1=caps1_n_caps, out_n2=10, n2_dim=16, n1_dim=8]
# caps1_output_tiled:[batch, caps1_n_caps=6*6*32, 10, 8, 1]
# caps2_output_predicted:[batch, caps1_n_caps=6*6*32, out_n2=10, n2_dim=16, 1]
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted") # 最内层的2维进行矩阵相乘
#
b= tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],dtype=np.float32, name="raw_weights") # [batch, caps1_n, caps2_n, 1,1]
c= tf.nn.softmax(b, dim=2, name="routing_weights") #

# 第一轮迭代
weighted_predictions = tf.multiply(c, caps2_predicted, name="weighted_predictions")
s = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
v = squash(s, axis=-2, name="caps2_output_round_1")

# 第二轮迭代
v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1], name="caps2_output_round_1_tiled")
agreement = tf.matmul(caps2_predicted, v_tiled, transpose_a=True, name="agreement")
b= tf.add(b, agreement, name="raw_weights_round_2")
c= tf.nn.softmax(b, dim=2, name="routing_weights_round_2")
weighted_predictions = tf.multiply(c, caps2_predicted, name="weighted_predictions_round_2")
s = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum_round_2")
v = squash(s, axis=-2, name="caps2_output_round_2")

#第3轮迭代
v_tiled = tf.tile(v, [1, caps1_n_caps, 1, 1, 1], name="caps2_output_round_2_tiled")
agreement = tf.matmul(caps2_predicted, v_tiled, transpose_a=True, name="agreement")
b= tf.add(b, agreement, name="raw_weights_round_3")
c= tf.nn.softmax(b, dim=2, name="routing_weights_round_3")
weighted_predictions = tf.multiply(c, caps2_predicted, name="weighted_predictions_round_3")
s = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum_round_3")
v = squash(s, axis=-2, name="caps2_output_round_3")

# 间隔损失
m_plus= 0.9
m_minus= 0.1
lambda_= 0.5
T= tf.one_hot(y, depth=caps2_n_caps, name="T")
v_norm= tf.norm(v, axis=-2, keep_dims=True, name="caps2_output_norm")
print("v_norm shape:",v_norm.shape) # (?, 1, 10, 1, 1)
y_prob = tf.reshape(v_norm,[-1, 10])
y_pred = tf.argmax(v_norm, axis=1)
print("y_pred:", y_pred.shape)

FP_raw= tf.square(tf.maximum(0., m_plus - v_norm), name="FP_raw")
FP= tf.reshape(FP_raw, shape=(-1, 10), name="FP")
FN_raw= tf.square(tf.maximum(0., v_norm - m_minus), name="FN_raw")
FN= tf.reshape(FN_raw, shape=(-1, 10), name="FN")
L= tf.add(T * FP, lambda_ * (1.0- T) * FN, name="L")
margin_loss= tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

# mask机制
mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
reconstruction_targets = tf.cond(mask_with_labels, lambda:y, lambda:y_pred, name="reconstruction_targets")
reconstruction_mask = tf.one_hot(reconstruction_targets, depth=caps2_n_caps, name="reconstruction_mask")
reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [-1, 1, caps2_n_caps, 1, 1], name="reconstruction_mask_reshaped")
reconstruction_mask_reshaped = tf.tile(reconstruction_mask_reshaped, [1,1,1,16, 1])
print("v shape:{} reconstruction_mask_reshape:{}".format(v.shape, reconstruction_mask_reshaped))
caps2_output_masked = tf.multiply(v, reconstruction_mask_reshaped, name="caps2_output_masked")

#解码器
n_hidden1 = 512
n_hidden2 = 1024
n_output = 28* 28
decoder_input = tf.reshape(caps2_output_masked, [-1, caps2_n_caps * caps2_n_dims], name="decoder_input")
with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, n_hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid, name="decoder_output")

# 重构损失
X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
squared_difference = tf.square(X_flat - decoder_output, name="squared_difference")
reconstruction_loss = tf.reduce_sum(squared_difference, name="reconstruction_loss")

# 最终损失
alpha= 0.0005
loss= tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

# 全局初始化
#init= tf.global_variables_initializer()
saver= tf.train.Saver()
# 计算精度
correct= tf.equal(y, y_pred, name="correct")
accuracy= tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
# 用 Adam 优化器
optimizer= tf.train.AdamOptimizer()
training_op= optimizer.minimize(loss, name="training_op")

N, width, height = X_train.shape
batch = 5
iter_num = N//batch
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iter in range(0, iter_num):
        index= iter * batch
        _, loss_, accuracy_ = sess.run([training_op, loss, accuracy], feed_dict={
           X: np.expand_dims(X_train[index:index+batch, :,:], axis=-1),
           y: y_train[index:index+batch],
        })
        print("iter:{} loss:{} accuracy:{}".format(iter, loss_, accuracy_))

