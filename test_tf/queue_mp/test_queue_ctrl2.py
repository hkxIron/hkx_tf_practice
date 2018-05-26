import tensorflow as tf
import numpy as np
import multiprocessing as mp

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

# class SingleProcess(mp.Process):
#     def __init__(self,name):
#         super(SingleProcess, self).__init__(name=name)

class QueueCtrl(object):

    def __init__(self):
        self.processes = []
        self.queues = []
        # self.queue = mp.Queue(1000)
        # self.X = tf.placeholder(tf.int64)
        # self.Y = tf.placeholder(tf.int64)
        # self.queue = tf.RandomShuffleQueue(dtypes=[tf.int64, tf.int64],
        #                                    capacity=100,
        #                                    min_after_dequeue=20)

        # self.enqueue_op = self.queue.enqueue([self.X, self.Y])


    def process_main(self,queue):
        """Cycle through the dataset until the main process says stop."""
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        train_fs = open(data_train, 'r')

        X = tf.placeholder(tf.int64)
        Y = tf.placeholder(tf.int64)
        enqueue_op = queue.enqueue([X, Y])
        print("process_main:")
        #while True:
        while not coord.should_stop():
            X_, Y_ = get_batch(train_fs)
            if not Y_: #We're at the end of the file
                train_fs = open(data_train, 'r')
                X_, Y_ = get_batch(train_fs)
                #queue.put([X_,Y_])
                print("queue put",[X_,Y_])
            #queue.enqueue([X_,Y_])
            #self.queue.enqueue([self.X,self.Y])
            sess.run(enqueue_op, feed_dict={X:X_, Y:Y_})

    def get_batch_from_queue(self):
        """
        Return one batch
        """
        #for process in self.processes:
        #return self.queue.get()
        #return self.queue.dequeue()
        for queue in self.queues:
            if queue.size()>0:
                return queue.dequeue()
            else:
                return [tf.constant(1),tf.constant(2)]

    def start_processes(self, num_processes=2):
        """Start the processes"""
        for _ in range(num_processes):
            queue = tf.RandomShuffleQueue(dtypes=[tf.int64, tf.int64],
                                        capacity=100,
                                        min_after_dequeue=20)
            #t = threading.Thread(target=self.thread_main, args=(sess, coord))
            t = mp.Process(target=self.process_main,args = (queue,))
            t.daemon = True
            t.start()
            self.processes.append(t)
            self.queues.append(queue)
        return self.processes

queue_ctrl = QueueCtrl()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
coord = tf.train.Coordinator()
tf.train.start_queue_runners(sess=sess, coord=coord)
my_thread = queue_ctrl.start_processes(num_processes=2)

# Then we build a dummy graph:
X_, Y_ = queue_ctrl.get_batch_from_queue()
output = Y_ * tf.reduce_sum(X_)

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(100):
    out = sess.run(output)
    print("Iter: %d, output: %d, ele num in queue: %d"
              % (i, out, sess.run(len(queue_ctrl.queues))))

coord.request_stop()
for _ in range(len(my_thread)): #if the queue is full at that time then the threads won't see the coord.should_stop
    _ = sess.run([output])

coord.join(my_thread, stop_grace_period_secs=10)
sess.close()



