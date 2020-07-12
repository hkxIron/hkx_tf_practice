import tensorflow as tf
import numpy as np

"""
tf.gather:
其中， params must be at least rank axis + 1，axis默认为0。类似于数组的索引，
可以把向量中某些索引值提取出来，得到新的向量，适用于要提取的索引为不连续的情况。
"""
a = tf.Variable([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15]])
index_a = tf.Variable([0, 2]) # 从第0维上取出第0个元素,以及第2个元素

b = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
index_b = tf.Variable([2, 4, 6, 8])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # gather中input的最后一个维度与output最后一个维度相同
    print("axis=0:\t", sess.run(tf.gather(a, index_a)))  # 获取行数为index_a的子数组
    print("axis=1:\t", sess.run(tf.gather(a, index_a, axis=1)))  # 列获取列数为index_a的子数组
    print("gather_b:\t", sess.run(tf.gather(b, index_b)))  # 当数组为一维时

'''
axis=0:	
[[ 1  2  3  4  5]
 [11 12 13 14 15]]
 
axis=1:	
[[ 1  3]
 [ 6  8]
 [11 13]]
 
gather_b: [3 5 7 9]
'''



"""
tf.gather_nd(params,indices,name=None,batch_dims=0)
返回值：根据indices的具体索引，取出params对应位置的值。
"""
a = tf.Variable([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15]])
index_a = tf.Variable([[0, 2], # 从axis=0取出第0个元素以及axis=1的第2个元素:3
                       [0, 4], # 从axis=0取出第0个元素以及axis=1的第4个元素:5
                       [2, 2]]) # 从axis=0取出第2个元素以及axis=1的第2个元素:13

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("gather_nd:", sess.run(tf.gather_nd(a, index_a)))
#  [ 3  5 13]



"""
应用
"""

# 修改二维数组(x,y)处的值
def set_value_first(matrix, x, y, val):
    # 提取出要更新的行
    row = tf.gather(matrix, x)
    # 构造这行的新数据
    new_row = tf.concat([row[:y], [val], row[y + 1:]], axis=0)
    # 使用 tf.scatter_update 方法进行替换
    matrix1 = tf.scatter_update(matrix, x, new_row)
    return matrix1

matrix = tf.Variable(tf.ones([3, 4]))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    matrix_first = set_value_first(matrix, 1, 2, 5.)
    print("matrix_first:\n", sess.run(matrix_first))
"""
matrix_first:
 [[1. 1. 1. 1.]
 [1. 1. 5. 1.]
 [1. 1. 1. 1.]]
"""

# 获取“数组”指定索引的值
arr = tf.constant([[1, 2, 3, 4, 5, 5, 7, 8, 9, 10],
                   [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                   [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]])
row = tf.range(4) # [0,1,2,3]
colum = tf.constant([2, 3, 4, 5]) # [2,3,4,5]

# tf.stack和tf.unstack的使用，详见:https://www.jianshu.com/p/25706575f8d4
ss = tf.stack([row, colum], axis=0)  # 构成[[0,1,2,3],[2,3,4,5]]
indexs = tf.unstack(ss, axis=1)  # 构成[[0,2],[1,3],[2,4],[3,5]]
newarr = tf.gather_nd(arr, indexs)

with tf.Session() as sess:
    print("newarr:",sess.run(newarr))

'''
[ 3 14 25 36]
'''

