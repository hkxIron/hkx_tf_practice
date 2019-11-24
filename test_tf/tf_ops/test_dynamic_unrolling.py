import tensorflow as tf
#import tensorflow.nn.rnn_cell.BasicRNNCell as BasicRNNCell

def test_dynamic():

    #0, 1 || 1, 2, 3, 5, 8, 13, 21
    #class fibonacci_core(BasicRNNCell):
    class fibonacci_core(object):
        def __init__(self):

            self.output_size = 1
            self.state_size = tf.TensorShape([1, 1])

    def __call__(self, input, state):
        return state[0] + state[1], (state[1], state[0] + state[1]) # output, next_state

    def zero_state(self, batch_size, dtype):
        return (tf.zeros((batch_size, 1), dtype=dtype), #state[0]
                tf.ones((batch_size, 1), dtype=dtype)) # state[1]

    def initial_state(self, batch_size, dtype):
        return zero_state(self, batch_size, dtype)

    inputs = tf.reshape(tf.range(10), [10, 1, 1])
    fibonacci_core_obj = fibonacci_core()
    fib_seq = tf.nn.dynamic_rnn(
        cell=fibonacci_core_obj,
        inputs=inputs,
        dtype=tf.float32,
        time_major=True)

    with tf.train.MonitoredSession() as sess:
        print(sess.run(fib_seq)) # (1,x 2, 3, 5, 8, 13, 21, 34, 55, 89)
        """
        我的测试发现,并不能运行
        TypeError: The argument 'cell' is not an RNNCell: either 'zero_state' or 'get_initial_state' method is required, 
        is not callable.
        """

test_dynamic()