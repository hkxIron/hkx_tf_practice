import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.random.set_random_seed(0)
embed=np.random.random((3,4))
x=np.random.random((3,2,4))

logit_np_ein = np.einsum("ntd,dk->ntk", x, np.transpose(embed))

input = tf.Variable(x) # [N, T , embed]
embedding = tf.Variable(embed) # [vocab, embed]
weights = tf.transpose(embedding) # (d_model, vocab_size)


# input:[N,T,embed], weights[d_model, vocab_size]
# logit_mat = tf.matmul(input, weights) # error!tf里没有batch matmul,必须要求tensor维度为2
input_shape = tf.shape(input)
#
input2 = tf.reshape(input,[-1, input_shape[-1]])
# [N*T, vocab_size]
logit_mat2 = tf.matmul(input2, weights)
logit_mat = tf.reshape(logit_mat2, [input_shape[0], input_shape[1], tf.shape(weights)[-1]])
# [N,T, vocab_size]
logit_tf_ein = tf.einsum("ntd,dk->ntk", input, weights)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
[mat_out, ein_tf_out]=sess.run([logit_mat, logit_tf_ein])

print("mat_out:", mat_out)
print("ein_tf_out:", ein_tf_out)
print("np_ein:", logit_np_ein)

"""
mat_out: [[[1.06402062 0.94727625 1.00463884]
           [1.54967637 1.66271466 1.41497592]]

          [[1.81208875 1.82876048 2.02767131]
           [1.12371763 1.36858855 0.97247593]]

          [[1.16429057 1.29511811 1.28082797]
           [1.00474438 1.11941126 0.9990769 ]]]
ein_tf_out: [[[1.06402062 0.94727625 1.00463884]
              [1.54967637 1.66271466 1.41497592]]

             [[1.81208875 1.82876048 2.02767131]
              [1.12371763 1.36858855 0.97247593]]

             [[1.16429057 1.29511811 1.28082797]
              [1.00474438 1.11941126 0.9990769 ]]]
np_ein: [[[1.06402062 0.94727625 1.00463884]
          [1.54967637 1.66271466 1.41497592]]

         [[1.81208875 1.82876048 2.02767131]
          [1.12371763 1.36858855 0.97247593]]

         [[1.16429057 1.29511811 1.28082797]
          [1.00474438 1.11941126 0.9990769 ]]]
"""
