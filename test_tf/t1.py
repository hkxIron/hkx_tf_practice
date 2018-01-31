# coding: utf-8
from __future__ import print_function
import tensorflow as tf

def t1():
    a = tf.random_normal((100, 100))
    b = tf.random_normal((100, 500))
    c = tf.matmul(a, b)
    sess = tf.InteractiveSession()
    res = sess.run(c)
    print(res)

    print("------------------")
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(node1, node2)

    print("after run session")
    sess = tf.Session()
    print(sess.run([node1, node2]))


    node3 = tf.add(node1, node2)
    print("node3:", node3)
    print("sess.run(node3):", sess.run(node3))

    print("placeholder")
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)


    print("adder_node: ", sess.run(adder_node, {a: 3, b: 4.5}))
    print("adder_node: ", sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

    add_and_triple = adder_node * 3.
    print("add_and_triple: ", sess.run(add_and_triple, {a: 3, b: 4.5}))


def t2():
    sess = tf.Session()
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W*x + b
    init = tf.global_variables_initializer()
    print("tf session:", sess.run(init)) # 只是初始化，并没有任何变量返回
    res = sess.run(linear_model, {x: [1, 2, 3, 4]})
    print("linear model:", res)


    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    # sess.run([fixW, fixb])
    print(sess.run([loss, fixW, fixb], {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    # print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    sess.run(init) # reset values to incorrect defaults.
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})


def t3():
    dist = tf.Variable(tf.random_normal([4,3]) ,dtype=tf.float32)
    # dist = tf.random_normal((4, 3))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # 只是初始化，并没有任何变量返回
    pred_value, pred_idx = tf.nn.top_k(dist, 1)  # 返回最大值以及它们的indices
    dist_run,pred_value_run,pred_idx_run = sess.run([dist,pred_value,pred_idx])
    print("dist_run: ",dist_run," pred_value_run: ",pred_value_run, " pred_idx_run: ",pred_idx_run)

def main():
    # t1()
    # t2()
    t3()

if __name__ == "__main__":
   main()

