import tensorflow as tf
import numpy as np
import threading

data_train = "../data/test.txt"

with open(data_train, 'w') as out_stream:
    out_stream.write("""[1,2,3,4,5,6]|1\n[1,2,3,4]|2\n[1,2,3,4,5,6]|0\n[1,2,3,4,5,6]|1\n[1,2,5,6]|1\n[1,2,5,6]|0""")

def get_batch(fs):
    line = fs.readline()
    X, Y = line.split('|')
    X = eval(X)
    Y = eval(Y)
    return X, Y

tf.reset_default_graph()#Reset the graph essential to use with jupyter else variable conflicts

class QueueCtrl(object):

    def __init__(self):
        self.X = tf.placeholder(tf.int64)
        self.Y = tf.placeholder(tf.int64)
        self.queue = tf.RandomShuffleQueue(dtypes=[tf.int64, tf.int64],
                                           capacity=100,
                                           min_after_dequeue=20)
        self.enqueue_op = self.queue.enqueue([self.X, self.Y])

    def thread_main(self, sess, coord):
        """Cycle through the dataset until the main process says stop."""
        train_fs = open(data_train, 'r')
        while not coord.should_stop():
            X_, Y_ = get_batch(train_fs)
            if not Y_: #We're at the end of the file
                train_fs = open(data_train, 'r')
                X_, Y_ = get_batch(train_fs)
            #self.queue.enqueue([self.X,self.Y])
            sess.run(self.enqueue_op, feed_dict={self.X:X_, self.Y:Y_})

    def get_batch_from_queue(self):
        """
        Return one batch
        """
        return self.queue.dequeue()

    def start_threads(self, sess, coord, num_threads=2):
        """Start the threads"""
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=self.thread_main, args=(sess, coord))
            t.daemon = True
            t.start()
            threads.append(t)
        return threads

# Then we build a dummy graph:
queue_ctrl = QueueCtrl()
X_, Y_ = queue_ctrl.get_batch_from_queue()
output = Y_ * tf.reduce_sum(X_)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess, coord=coord)
my_thread = queue_ctrl.start_threads(sess, coord, num_threads=6)

for i in range(100):
    out = sess.run(output)
    print("Iter: %d, output: %d, ele num in queue: %d"
              % (i, out, sess.run(queue_ctrl.queue.size())))

coord.request_stop()
for _ in range(len(my_thread)): #if the queue is full at that time then the threads won't see the coord.should_stop
    _ = sess.run([output])

coord.join(my_thread, stop_grace_period_secs=10)
sess.close()



