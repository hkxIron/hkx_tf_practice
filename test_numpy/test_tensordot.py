import numpy as np
import tensorflow as tf
np.random.seed(0)

"""
解释：
(1,2) 是对A而言，不是取第1，2轴，而是除去1,2 轴，所以要取的是第0轴
(0,1) 是对B而言，不是取第0，1轴，而是除去0,1 轴，所以要取的是第2轴

以上两句是精华

A的形状是(3,4,5)，第0轴上有3个元素，取法上面讲了；B的形状(4,5,2)，第2轴上有2个元素，所以结果形状是(3,2)

Tensordot 的作用就是把取出的子数组做点乘操作，即是 np.sum(a*b) 操作。
我们来验证一下，上述的说法看结果形状（3，2）的第一个元素：A第0轴上第一个元素与B第2轴上的第一个元素点乘。


"""
def matrix_equal(A, B):
    return np.sum(np.abs(A - B)) == 0

def f1():
    X = np.random.randint(0, 9, (3, 4, 5))
    print(X)
    # 取第0轴的第0个元素
    print("axis 0:")
    print(X[0,:,:]) # 或者X[0]

    # 取第1轴的第0个元素
    print("axis 1:")
    print(X[:,0,:])

    # 取第2轴的第0个元素
    print("axis 1:")
    print(X[:,:,0])

    A = np.random.randint(0, 9, (3, 4, 5))
    B = np.random.randint(0, 9, (4, 5, 2))

    ab_dot=np.tensordot(A, B, [(1, 2), (0, 1)])

    ab = np.sum(A[0,:,:] * B[:,:,0])
    print("ab:\n", ab)
    print("ab_dot:\n", ab_dot)

    x = tf.constant(A)
    y = tf.constant(B)
    xy = tf.tensordot(x, y, axes=[[1,2], [0,1]])
    print("tf tensor dot:")
    with tf.Session() as sess:
        xy_dot = sess.run(xy)
    print(xy_dot)
    assert matrix_equal(xy_dot, ab_dot)

def f2():
    np.random.seed(0)
    A = np.random.randint(0, 10, (3, 4, 5))
    B = np.random.randint(0, 10, (5, 6))
    C = np.einsum('ijk,kl->ijl', A, B)
    #C_tf = tf.einsum('ijk,kl->ijl', A, B)
    A_tf = tf.constant(A)
    B_tf = tf.constant(B)
    C_tf = tf.einsum('ijk,kl->ijl', A_tf, B_tf)
    print("numpy einsum C shape:{} \nC:{}".format(C.shape, C))
    with tf.Session() as sess:
       C_out = sess.run(C_tf)
    print("tf einsum C shape:{} \nC:{}".format(C_out.shape, C_out))
    assert matrix_equal(C, C_out)

#f1()
f2()
