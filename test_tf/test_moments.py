import tensorflow as tf

tf.random.set_random_seed(0)
inputs = tf.Variable(tf.random_normal([3,4,5]))
mean, variance = tf.nn.moments(inputs, axes=[-1], keep_dims=True) # 在最后一个维度上计算mean, var

sess = tf.Session()
sess.run(tf.global_variables_initializer())

[mean_out, var_out] = sess.run([mean, variance])

print("mean_out shape:", mean_out.shape)
print("var_out shape:", var_out.shape)

print("mean_out:", mean_out)
print("var_out:", var_out)


"""
mean_out shape: (3, 4, 1)
var_out shape: (3, 4, 1)
mean_out: [[[-0.41930795]
  [ 0.24208717]
  [-0.080189  ]
  [ 0.27765483]]

 [[-0.31298566]
  [-0.21449816]
  [-0.94180334]
  [-0.12509248]]

 [[-0.5256568 ]
  [ 0.02891991]
  [ 0.2737916 ]
  [ 0.003728  ]]]
var_out: [[[0.40091902]
  [0.4730677 ]
  [0.49321014]
  [0.70209324]]

 [[0.5181615 ]
  [0.21795149]
  [0.6249728 ]
  [0.6099998 ]]

 [[0.4274796 ]
  [0.5061819 ]
  [1.2654319 ]
  [0.40912628]]]

Process finished with exit code 0
"""