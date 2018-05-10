from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

class TensorflowTest(test.TestCase):

    def testNormal(self):
        super(TensorflowTest,self).setUp()
        arr = np.array([
            [1, 2, 3], # 1/(1+4+9) = 1/sqrt(14)=0.2672612419124244
            [4, 5, 6],
            [7, 8, 9]
        ])
        arr_normed = arr / np.expand_dims(np.sqrt(np.sum(arr * arr, axis=1)), axis= -1)
        print("normed:", arr_normed)


        with self.test_session(use_gpu=False):
            tf_arr = constant_op.constant(arr, dtype=tf.float32)
            tf_arr_normed = tf_arr / tf.expand_dims(tf.sqrt(tf.reduce_sum(tf_arr * tf_arr, axis=1)), -1)

            print("tf_normed: \n", tf_arr_normed.eval()) # 可以验证gather 与embedding lookup的功能是一样的
