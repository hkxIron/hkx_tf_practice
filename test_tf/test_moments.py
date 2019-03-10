import tensorflow as tf

inputs = tf.Variable(tf.random_normal([3,4,5]))
mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

[mean_out, var_out] = sess.run([mean, variance])

print("mean_out shape:", mean_out.shape)
print("var_out shape:", var_out.shape)

print("mean_out:", mean_out)
print("var_out:", var_out)
