import os
import tensorflow as tf
import pathos.multiprocessing as pmp


def f(graph, queue,i):
    # g = tf.Graph()
    with graph.as_default():
        # q = tf.FIFOQueue(capacity=100, dtypes=[tf.string], shapes=[[]],
        #                  shared_name="shared_q", name="q")
        a = tf.constant("Hello")
        b = queue.enqueue(a)
        q_size = queue.size()
    with tf.Session(graph=graph) as sess:
        for n in range(i):
            sess.run(b)
            print("sub function:", sess.run(q_size))
    return

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pool = pmp.Pool(2)
    graph = tf.Graph()
    with graph.as_default():
        queue = tf.FIFOQueue(capacity=100, dtypes=[tf.string], shapes=[[]],
                             shared_name="shared_q", name="q")
        qsz = queue.size()
        pool.starmap(f, [(graph,queue, 1), (graph,queue, 2)])
        with tf.Session(graph=graph) as sess:
            for i in range(3):
                print("main:",sess.run(qsz))
        #os.unsetenv("CUDA_VISIBLE_DEVICES")
