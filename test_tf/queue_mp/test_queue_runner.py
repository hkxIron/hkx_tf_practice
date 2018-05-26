#encoding=utf-8
import numpy as np
import tensorflow as tf
batch_size = 2
#随机产生一个2*2的张量
example = tf.random_normal([2,2])
#创建一个RandomShuffleQueue，参数意义参见API

q = tf.RandomShuffleQueue(
    capacity=1000,
    min_after_dequeue=0,
    dtypes=tf.float32,
    shapes=[2,2])
#enqueue op，每次push一个张量
#enq_op = q.enqueue_many([example for _ in range(2) ])
enq_op = q.enqueue(example)
#enq_op = q.enqueue(example)
#dequeue op, 每次取出batch_size个张量
xs = q.dequeue_many(batch_size)
#创建QueueRunner，包含4个enqueue op线程
qr = tf.train.QueueRunner(q, [enq_op]*4)
coord = tf.train.Coordinator()
sess = tf.Session()
#启动QueueRuner，开始线程
enq_threads = qr.create_threads(sess, coord=coord, start=True)
for i in range(10):
    if coord.should_stop():
        break
    print('step:', i, sess.run(xs)) #打印结果
coord.request_stop()
coord.join(enq_threads)
