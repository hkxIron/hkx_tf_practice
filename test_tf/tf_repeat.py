import tensorflow as tf
import numpy as np

y= np.arange(0, 8).reshape((4,2))
print("y:", y)
sess= tf.Session()
yp = tf.constant(y)

def f1():
    idx = tf.range(len(y))
    idx = tf.reshape(idx, [-1, 1])    # Convert to a len(yp) x 1 matrix.

    K = len(y) # 重复次数
    idx = tf.tile(idx, [1, K])  # Create multiple columns.
    idx = tf.reshape(idx, [-1])       # Convert back to a vector.

    jdx = tf.range(len(y))
    jdx = tf.tile(jdx, [len(y)])

    x_repeat = tf.gather(yp, idx)
    x_tile = tf.gather(yp, jdx)
    #s = x_repeat - x_tile

    print("x_repeat:", sess.run(x_repeat))
    print("x_tile:", sess.run(x_tile))


"""
def f2():
    value = np.arange(len(y))  # what to repeat
    repeat_count = len(y)  # how many times
    repeated = tf.stack([value for i in range(repeat_count)], axis=1)
    print("repeat:", sess.run(repeated))
"""

# 或者利用keras接口
def f3():
    repeated = tf.keras.backend.repeat_elements(yp, rep=3, axis=0)
    print("repeat:", sess.run(repeated))

f1()
f3()
