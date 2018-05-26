import multiprocessing

import tensorflow as tf

def f(x):
    session = tf.Session()
    a = tf.Variable(x, name='a')
    b = tf.Variable(100, name='b')
    c = tf.multiply(a, b, name='c')
    session.run(tf.global_variables_initializer())

    out = session.run(c)
    print("OK: %s" % out)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Comment me out to hang
    f(0)
    multiprocessing.Pool().map(f, range(10))