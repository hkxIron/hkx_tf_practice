import tensorflow as tf

"""Returns a mask tensor representing the first N positions of each cell.

def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):

If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with

```
mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
```
# 可以看出 sequence_mask：会在最后一维增加一个维度,返回bool值的矩阵
Examples:

tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
                                #  [True, True, True, False, False],
                                #  [True, True, False, False, False]]

tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
                                  #   [True, True, True]],
                                  #  [[True, True, False],
                                  #   [False, False, False]]]
"""

seq_mask = tf.sequence_mask(lengths=[1, 3, 2], maxlen=5)
sess = tf.Session()
print("seq_mask: ", sess.run(seq_mask)) # 3*5

seq_mask = tf.sequence_mask(lengths=[[1, 3],[2, 0]], maxlen=5)
sess = tf.Session()
print("seq_mask: ", sess.run(seq_mask)) # 3*5
