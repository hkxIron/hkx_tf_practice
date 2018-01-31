import tensorflow as tf

def q1():
    #创建一个先进先出队列
    q = tf.FIFOQueue(2, "int32")
    init = q.enqueue_many(([0, 10],))
    x = q.dequeue()
    y = x + 1
    q_inc = q.enqueue([y])
    with tf.Session() as sess:
        init.run()
        for _ in range(5):
            v, _ = sess.run([x, q_inc])
            print (v)

def q2():
    #queue = tf.FIFOQueue(100,tf.int64)
    queue = tf.FIFOQueue(100, "float")
    enqueue_op = queue.enqueue([tf.random_normal([1])])
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)
    tf.train.add_queue_runner(qr)
    out_tensor = queue.dequeue()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for _ in range(3): print(sess.run(out_tensor)[0])
        coord.request_stop()
        coord.join(threads)
q2()