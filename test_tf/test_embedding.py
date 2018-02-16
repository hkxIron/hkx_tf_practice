from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test

def embeddingPad(w, input, padding=False, sum=False, name=None):
    if not padding:
        y = tf.gather(w, input, name=name)
    else:
        mask = tf.cast(tf.cast(input, tf.bool), w.dtype) # 由于tf里会将0当成false,其余非0当作true
        y = tf.gather(w, input)
        y = tf.multiply(y, tf.expand_dims(mask, -1), name=name) #这个是向量逐元素相乘
    return y

class EmbeddingTest(test.TestCase):

    def testEmbeddingPadOnly(self):
        super(EmbeddingTest,self).setUp()
        with self.test_session(use_gpu=False):
            embedding = constant_op.constant([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ], dtype=tf.int32)
            ids=np.array([0, 2, 1, 0]) # 相当于将那些没有选出来的embedding全都padding成0，但要求ids也要pad，无值的地方都填0
            mask_pre=tf.cast(ids, tf.bool)
            mask = tf.cast(mask_pre, embedding.dtype)
            result_pre = tf.gather(embedding, ids)
            result = tf.multiply(result_pre, tf.expand_dims(mask, -1))
            print("mask_pre:",mask_pre.eval())
            print("mask:",mask,mask.eval())
            print("result_pre:",result_pre.eval())
            print("result:",result.eval())

            """
            mask_pre: [False  True  True False]
            mask: Tensor("Cast_1:0", shape=(4,), dtype=int32, device=/device:CPU:0) [0 1 1 0]
            result_pre: 
            [[1 2 3]
             [7 8 9]
             [4 5 6]
             [1 2 3]]

            result: 
            [[0 0 0]
             [7 8 9]
             [4 5 6]
             [0 0 0]]
            """

    def testEmbeddingPad(self):
        super(EmbeddingTest,self).setUp()
        with self.test_session(use_gpu=False):
            embedding = constant_op.constant([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ], dtype=tf.int32)
            ids=np.array([0, 2, 1, 0])
            result_no_pad=embeddingPad(embedding,ids).eval()
            print("no pad:",result_no_pad)

            result_pad=embeddingPad(embedding,ids).eval()
            print("pad:",result_pad) # 并没有发现二者有什么区别

    def testEmbedding(self):
        super(EmbeddingTest,self).setUp()
        with self.test_session(use_gpu=False):
            embedding = constant_op.constant([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],dtype=tf.int32)
            ids=np.array([0, 2, 1, 0])
            result = tf.nn.embedding_lookup(embedding,ids).eval()
            expect_result= constant_op.constant([[ 1,  2,  3], [ 7, 8,  9], [ 4,  5,  6], [ 1,  2, 3]]).eval()
            print("result: \n", result)
            self.assertAllEqual(result,expect_result)

            gather_ids = np.array([0, 2, 1, 0])
            result = tf.nn.embedding_lookup(embedding,gather_ids).eval()
            print("gather_result: \n", result) # 可以验证gather 与embedding lookup的功能是一样的
            self.assertAllEqual(result,expect_result)

            """
            由于输入ids是2d，所以result是3d,即总的维数是embedding的维数加上ids的维数
            ids的每个元素均代码embedding里的一行，即embedding[i,:],
            [
              [embedding[0,:],embedding[2,:]],
              [embedding[1,:],embedding[0,:]]
            ]
            
            """
            ids=np.array([[0, 2],[ 1, 0]])
            print("ids in 2D array: ",ids)
            result = tf.nn.embedding_lookup(embedding,ids).eval()
            #result = tf.gather(embedding,ids).eval() 与embedding_lookup是一样的功能
            print("result in 2D: \n", result)
            expect_result= constant_op.constant([[[ 1,  2,  3], [ 7, 8,  9]],
                                                 [[ 4,  5,  6], [ 1,  2, 3]]]).eval()
            self.assertAllEqual(result,expect_result)

