import tensorflow as tf

g1=tf.get_default_graph()
g2=tf.Graph()

with g1.as_default():
    a=tf.multiply(3,4)

with g2.as_default():
    b=tf.add(3,4)

sess=tf.Session(graph=tf.get_default_graph())
print("a in g1(default):",sess.run(a))

sess=tf.Session(graph=g2)
print("a in g2(default):",sess.run(b))
