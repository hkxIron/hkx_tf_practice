import torch
import tensorflow as tf
import numpy as np
import random

print("tf version:", tf.__version__)
np.random.seed(0)
tf.random.set_random_seed(0)
torch.manual_seed(0) # cpu
torch.cuda.manual_seed(0) #gpu
random.seed(0)

def dynamic_net_torch():
    N, D, H = 3,4,5
    x = torch.randn(N, D, requires_grad=True)
    w1 = torch.randn(D, H)
    w2 = torch.randn(D, H)

    z =10
    if z>0:
        y=x.mm(w1)
    else:
        y=x.mm(w2)
    print("torch y:", y)

def dynamic_net_tf():
    N, D, H = 3, 4, 5
    x = tf.placeholder(tf.float32, shape=(N, D))
    z = tf.placeholder(tf.float32, shape=None)
    w1 = tf.placeholder(tf.float32, shape=(D, H))
    w2 = tf.placeholder(tf.float32, shape=(D, H))

    def f1():return tf.matmul(x, w1)
    def f2():return tf.matmul(x, w2)
    y= tf.cond(tf.greater(z,0), true_fn=f1, false_fn=f2)

    with tf.Session() as sess:
        values = {
            x:np.random.randn(N, D),
            z:10,
            w1:np.random.randn(D, H),
            w2:np.random.randn(D, H),
        }
        y_value = sess.run(y, feed_dict=values)
        print("tf y:", y_value)

dynamic_net_torch()
dynamic_net_tf()

def dynamic_loop_net_torch():
    T, D = 3,4
    y0 = torch.randn(D, requires_grad=True)
    x = torch.randn(T, D)
    w = torch.randn(D)

    y=[y0]
    for t in range(T):
        prev_y = y[-1]
        next_y = (prev_y + x[t]) *w
        y.append(next_y)
    print("torch loop y:", y[-1])

dynamic_loop_net_torch()

def dynamic_loop_net_tf():
    T, N, D = 3, 4, 5
    x = tf.placeholder(tf.float32, shape=(T, D))
    y0 = tf.placeholder(tf.float32, shape=(D, ))
    w = tf.placeholder(tf.float32, shape=(D, ))

    def f(prev_y, cur_x):
        return (prev_y + cur_x)*w

    y = tf.foldl(fn=f, elems=x, initializer=y0)

    with tf.Session() as sess:
        values = {
            x:np.random.randn(T, D),
            y0:np.random.randn(D),
            w:np.random.randn(D),
        }
        y_val = sess.run(y, feed_dict=values)
        print("tf loop y:", y_val)

dynamic_loop_net_tf()

# torch.onnx.export("")


