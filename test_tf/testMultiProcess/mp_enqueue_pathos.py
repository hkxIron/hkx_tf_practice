import os
import tensorflow as tf
import pathos.multiprocessing as pmp


def f(g, i):
    # g = tf.Graph()
    with g.as_default():
        q = tf.FIFOQueue(capacity=100, dtypes=[tf.string], shapes=[[]],
                         shared_name="shared_q", name="q")
        a = tf.constant("Hello")
        b = q.enqueue(a)
        c = q.size()
        with tf.variable_scope("",reuse=True):
            v=tf.get_variable(name="test_var")
            v+=1
    with tf.Session(graph=g) as sess:
        sess.run(tf.initialize_all_variables)
        for n in range(i):
            sess.run([b,v])
            print("sub function:",sess.run(c))
    return


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    p = pmp.Pool(2)
    g = tf.Graph()
    with g.as_default():
        q = tf.FIFOQueue(capacity=100, dtypes=[tf.string], shapes=[[]],
                         shared_name="shared_q", name="q")
        v=tf.get_variable(name="test_var",shape=[1,1],dtype=tf.float32,initializer=tf.ones_initializer)
        qsz = q.size()
        p.starmap(f, [(g, 1), (g, 2)])
        with tf.Session(graph=g) as sess:
            for i in range(3):
                print("main:",sess.run(qsz))
        #os.unsetenv("CUDA_VISIBLE_DEVICES")
