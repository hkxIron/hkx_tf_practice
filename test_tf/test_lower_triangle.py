import tensorflow as tf

tf.random.set_random_seed(0)
inputs = tf.Variable(tf.random_normal([4,3]))
tril = tf.linalg.LinearOperatorLowerTriangular(tril=inputs).to_dense()  # (T_q, T_k)

sess =tf.Session()
sess.run(tf.global_variables_initializer())
input_out,tril_out = sess.run([inputs, tril])
print("input:",input_out)
print("tril:",tril_out)
