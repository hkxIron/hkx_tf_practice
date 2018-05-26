import tensorflow as tf

g = tf.Graph()
class A:
    #g = tf.Graph()
    with g.as_default():
        x = tf.placeholder(tf.float32)

    s = tf.Session(graph=g)

    __call__ = lambda self,X: self.s.run(self.y, {self.x:X})

class B(A):
    with g.as_default():
        y = 2 * A.x


test = B()
print(test([1,1,2]))