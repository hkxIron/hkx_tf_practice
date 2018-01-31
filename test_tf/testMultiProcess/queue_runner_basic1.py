import tensorflow as tf

# We simulate some raw input data
# let's start with only 3 samples of 1 data point
x_input_data = tf.random_normal([3], mean=-1, stddev=4)

# We build a FIFOQueue inside the graph
# You can see it as a waiting line that holds waiting data
# In this case, a line with only 3 positions
q = tf.FIFOQueue(capacity=3, dtypes=tf.float32)

# We need an operation that will actually fill the queue with our data
# "enqueue_many" slices "x_input_data" along the 0th dimension to make multiple queue elements
enqueue_op = q.enqueue_many(x_input_data) # <- x1 - x2 -x3 |

# We need a dequeue op to get the next elements in the queue following the FIFO policy.
input = q.dequeue()
# The input tensor is the equivalent of a placeholder now
# but directly connected to the data sources in the graph

# Each time we use the input tensor, we print the number of elements left
# in the queue
input = tf.Print(input, data=[q.size()], message="Nb elements left:")

# fake graph: START
y = input + 1
# fake graph: END

# We start the session as usual
with tf.Session() as sess:
    # We first run the enqueue_op to load our data into the queue
    sess.run(enqueue_op)
    # Now, our queue holds 3 elements, it's full.
    # We can start to consume our data
    sess.run(y)
    sess.run(y)
    sess.run(y)
    # Now our queue is empty, if we call it again, our program will hang right here
    # waiting for the queue to be filled by at least one more datum
    sess.run(y)

