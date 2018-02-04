import tensorflow as tf
data = [
        [[1, 1, 1], [2, 2, 2]],
        [[3, 3, 3], [4, 4, 4]],
        [[5, 5, 5], [6, 6, 6]]
       ]
data_tensor =tf.convert_to_tensor(data,dtype=tf.int32)
print("data:",data, "data_tensor:",data_tensor)
x = tf.strided_slice(data,[0,0,0],[1,1,1])
with tf.Session() as sess:
    print(sess.run(x))
"""
[[[1]]]
"""
print("----------------")
# strided_slice是不包含end索引
x = tf.strided_slice(data,[0,0,0],[2,2,3]) # 0维是最外面那层括号，第1维是次外层括号，第2维是最里层括号
with tf.Session() as sess:
    print(sess.run(x))
"""
[
 [[1 1 1]
  [2 2 2]]

 [[3 3 3]
  [4 4 4]]
]
"""

# 当指定stride为[1,1,1]输出和没有指定无区别，可以判断默认的步伐就是每个维度为1
print("----------------")
x = tf.strided_slice(data,[0,0,0],[2,2,2],[1,2,1])
with tf.Session() as sess:
    print(sess.run(x))

"""
[[[1 1]]

 [[3 3]]]
"""

print("----------------")
# 当begin为正值，stride任意位置为负值，输出都是空的
x = tf.strided_slice(data, [1, -1, 0], [2, -3, 3], [1, -1, 1])
with tf.Session() as sess:
    print(sess.run(x))
"""
[[[4 4 4]
  [3 3 3]]]
"""
