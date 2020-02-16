import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import tensorflow as tf

# gather与embedding_lookup感觉是一样的
print("embedding lookup...")
vocab = tf.Variable(tf.random_normal([10, 2], seed=0))  # [vocab, embed]
lookedup_vocab = tf.nn.embedding_lookup(vocab, ids=[1, 3])  #id:[batch], [batch, embed],查找张量中的序号为1和3的
gathered_vocab = tf.gather(vocab, [1,3])

arg_max_tensor = tf.argmax(vocab, axis=-1, output_type=tf.int32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("vocab:")
    print(sess.run(vocab))
    print("arg_max_tensor:")
    print(sess.run(arg_max_tensor))
    print(arg_max_tensor.shape)
    print("lookedup vocab:")
    print(sess.run(lookedup_vocab))
    print("gather vocab:")
    print(sess.run(gathered_vocab))


print("\n\ngather...")
import tensorflow as tf
params = tf.range(0, 10)*10
#params = vocab
gathered = tf.gather(params, [2, 5, 8])
lookedup = tf.nn.embedding_lookup(params, [2, 5, 8])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("params:",sess.run(params))
    print("gatherd:", sess.run(gathered))
    print("lookedup:",sess.run(lookedup))
