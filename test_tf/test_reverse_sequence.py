# blog: https://blog.csdn.net/guotong1988/article/details/52767418
# paper: http://www.aclweb.org/anthology/N16-1030
import tensorflow as tf

"""
  This op first slices `input` along the dimension `batch_dim`, and for each
  slice `i`, reverses the first `seq_lengths[i]` elements along
  the dimension `seq_dim`.

按照官方给出的解释，
“此操作首先沿着维度batch_axis对input进行分割，并且对于每个切片 i，将前 seq_lengths 元素沿维度 seq_axis 反转”。
实际上通俗来理解，就是对于张量input中的第batch_axis维中的每一个子张量，在这个子张量的第seq_axis维上进行翻转， 翻转的长度为 seq_lengths 张量中对应的数值。  
其中seq_axis与batch_axis的维度都是相对于原始tensor

举个例子，
如果 batch_axis=0，seq_axis=1，则代表我希望每一行为单位分开处理， 对于每一行中的每一列进行翻转。
相反的，如果 batch_axis=1，seq_axis=0，则是以列为单位，对于每一列的张量，进行相应行的翻转。
回头去看双向RNN的源码，就可以理解当time_major这一属性不同时，time_dim 和 batch_dim 这一对组合的取值为什么恰好是相反的了。
"""

input = tf.constant([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
seq_lengths = tf.constant([1,2,3],tf.int64) # 每一次翻转长度分别为1,2,3.由于a是（3,3）维的，所以l中数值最大只能是3
x = tf.reverse_sequence(input, seq_lengths=seq_lengths, seq_axis = 1, batch_axis= 0) # 以行为单位进行翻转，翻转的是每一列的元素
y = tf.reverse_sequence(input, seq_lengths=seq_lengths, seq_axis = 0, batch_axis= 1) # 以列为单位进行翻转，翻转的是每一行的元素
with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))

"""
行内元素交换:每一行上的元素种类没有发生变化，但是从每一列来看，列的顺序分别翻转了前1，前2，前3个元素
因此有:
第一行:[1,2,3],前1个元素交换还是[1,2,3]
第二行:[4,5,6],前2个元素交换,为[5,4,6]
第三行:[7,8,9],前3个元素交换,为[9,8,7]
[[1 2 3]
 [5 4 6]
 [9 8 7]]

同理 
列内元素交换:每一列上的元素种类没有发生变化，但是从每一行来看，行的顺序分别翻转了前1，前2，前3个元素
[[1 5 9]
 [4 2 6]
 [7 8 3]]
"""