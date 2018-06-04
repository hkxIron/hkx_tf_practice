import numpy as np
import tensorflow as tf

a = tf.placeholder(tf.float32, shape=[None, 2])
k = tf.placeholder(tf.int32)

# 从二维矩阵中随机选取k个元素
y = tf.py_func(func=lambda x, s: np.random.choice(x.reshape(-1),s),inp=[a, k],Tout= tf.float32)

np.random.seed(0)
mat = np.random.random((4,2))

print("mat: ",mat,"\n")
with tf.Session() as sess:
    print(sess.run(y, {a: mat, k:5}))

# 直接使用np.random
idx = np.arange(0,5)
print("idx:",idx)
select_idx = np.random.choice(idx,2)
print(" select_idx:",select_idx)
print(" np.random:",mat[select_idx,:] )
