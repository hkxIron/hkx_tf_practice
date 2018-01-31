import tensorflow as tf
g = tf.Graph()
with g.as_default():
    # build graph...
    x = tf.constant(1)

sess = tf.Session(graph=g)
print(sess.run(x))
sess.close()

# Now we can create a new session with the same graph
sess2 = tf.Session(graph=g)
print(sess2.run(x))
sess2.close()