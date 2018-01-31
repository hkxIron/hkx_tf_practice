import tensorflow as tf

y = [tf.random_uniform([2,3]) for n in range(1,10)]
#y = [tf.constant(n) for n in range(1,10)]
#y = [tf.constant(range(n)) for n in range(1,10)]
print(y)

batched_data = tf.train.batch(
    tensors=[y],    #tensors=[y]
    batch_size=2,
    dynamic_pad=True,
    name="y_batch",
    enqueue_many=False
    #enqueue_many=False
)

# Run the graph
# tf.contrib.learn takes care of starting the queues for us
res = tf.contrib.learn.run_n({"y": batched_data}, n=1, feed_dict=None)

# Print the result
#print("Batch shape: {}".format(res[0]["y"].shape))
print(res[0]["y"])