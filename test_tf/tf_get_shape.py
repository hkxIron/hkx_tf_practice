import tensorflow as tf
import numpy as np


tensor = tf.constant([[[1, 1, 1],
                       [2, 2, 2]],
                      [[3, 3, 3],
                       [4, 4, 4]]])
sess = tf.Session()
print(sess.run(tf.shape(tensor)))  # [2, 2, 3], tf.shape返回的是tensor
print(tensor.get_shape())  # (2, 2, 3), 返回的是元组

np.random.seed(0)
x=tf.constant(np.random.random((3,4,1,5)))
y=tf.constant(np.random.random((3,1,4,5)))
v=tf.constant(np.random.random((5)))
z=tf.tanh(x+y)
t=v*z
print("x shape:", x.get_shape()) # (3,4,1,5)
print("y shape:", y.get_shape()) # (3,1,4,5)
print("z shape:", z.get_shape()) # (3,4,4,5)
print("t shape:", t.get_shape()) # (3,4,4,5)

print(sess.run([x]))
print(sess.run([y]))
print(sess.run([z]))

# hidden_features:[batch, 1,input_sequence_length, hidden_size * 2]
# y:[batch, input_sequence_length,1 , hidden_size* 2]
