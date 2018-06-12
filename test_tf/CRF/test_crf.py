# coding=utf8
"""
url:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf

CRF
The CRF module implements a linear-chain CRF layer for learning to predict tag sequences. This variant of the CRF is factored into unary potentials for every element in the sequence and binary potentials for every transition between output tags.

Usage
Below is an example of the API, which learns a CRF for some random data. The linear layer in the example can be replaced by any neural network.
"""
import numpy as np
import tensorflow as tf

# data settings
num_examples = 10
num_words = 20
num_features = 100
num_tags = 5

# 5 tags
# x shape = [10,20,100]
# random features.
np.random.seed(0)
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

# y shape = [10,20]
# Random tag indices representing the gold sequence.
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# 序列的长度
# sequence_lengths = [19,19,19,19,19,19,19,19,19,19]
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)

# Train and evaluate the model.
with tf.Graph().as_default():
    with tf.Session() as session:
        # Add the data to the TensorFlow gtaph.
        x_t = tf.constant(x)  # 观测序列
        y_t = tf.constant(y)  # 标记序列
        sequence_lengths_t = tf.constant(sequence_lengths)

        # Compute unary scores from a linear layer.
        # weights shape = [100,5]
        weights = tf.get_variable("weights", [num_features, num_tags])

        # matricized_x_t shape = [200,100]
        matricized_x_t = tf.reshape(x_t, [-1, num_features])

        # compute                           [200,100]      [100,5]   get [200,5]
        # 计算结果
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)

        #  unary_scores shape = [10,20,5]                  [10,20,5]
        unary_scores = tf.reshape(matricized_unary_scores, [num_examples, num_words, num_tags])
        # compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        #                                                shape      shape   [10,20,5]   [10,20]   [10]
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, y_t, sequence_lengths_t)

        viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(unary_scores, transition_params, sequence_lengths_t)
        # add a training op to tune the parameters.
        loss = tf.reduce_mean(-log_likelihood)

        # 定义梯度下降算法的优化器
        # learning_rate 0.01
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # train for a fixed number of iterations.
        session.run(tf.global_variables_initializer())

        ''' 
       #eg:
       In [61]: m_20
       Out[61]: array([[ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])

       In [62]: n_20
       Out[62]: array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

       In [59]: n_20<m_20
       Out[59]: array([[ True,  True,  True,  True,  True,  True,  True,  True,  True, True]], dtype=bool)

        '''
        # 这里用mask过滤掉不符合的结果
        mask = (np.expand_dims(np.arange(num_words), axis=0) < np.expand_dims(sequence_lengths, axis=1))

        ###mask = array([[ True,  True,  True,  True,  True,  True,  True,  True,  True, True]], dtype=bool)
        # 序列的长度
        total_labels = np.sum(sequence_lengths)

        print("mask:", mask)

        print("total_labels:", total_labels)
        for i in range(1000):
            # tf_unary_scores,tf_transition_params,_ = session.run([unary_scores,transition_params,train_op])
            tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op])
            if i % 100 == 0:
                '''
                false*false = false  false*true= false ture*true = true
                '''
                # 序列中预测对的个数
                correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)