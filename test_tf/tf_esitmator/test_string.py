import tensorflow as tf

string = ["hello world", "a b c", "hello hkx c"]
string_tensor = tf.convert_to_tensor(string)
with tf.Session() as sess:
    chars = tf.string_split(string, " ") # tensorflow 还是有op，可以支持 split,  返回SparseTensor
    print("string : ", sess.run(string_tensor))
    print("string newaxis: ", sess.run(string_tensor[tf.newaxis])) # 相当于插入一维
    print("string expand_dim 0: ", sess.run(tf.expand_dims(string_tensor, dim=0))) # 相当于插入一维
    print("string expand_dim -1: ", sess.run(tf.expand_dims(string_tensor, dim=-1))) # 相当于插入一维
    print("sparse tensor: ", sess.run(chars))
    print("sp values: ", sess.run(chars.values))

"""
string :  [b'hello world' b'a b c' b'hello hkx c']
string newaxis:  [[b'hello world' b'a b c' b'hello hkx c']]
string expand_dim 0:  [[b'hello world' b'a b c' b'hello hkx c']]
string expand_dim -1:  [[b'hello world']
 [b'a b c']
 [b'hello hkx c']]
sparse tensor:  SparseTensorValue(indices=array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1],
       [1, 2],
       [2, 0],
       [2, 1],
       [2, 2]], dtype=int64), values=array([b'hello', b'world', b'a', b'b', b'c', b'hello', b'hkx', b'c'], dtype=object), dense_shape=array([3, 3], dtype=int64))
sp values:  [b'hello' b'world' b'a' b'b' b'c' b'hello' b'hkx' b'c']
"""

sess = tf.Session()
string = ["hello world", "a b c", "hello hkx c"]
string_bucket = tf.string_to_hash_bucket(string, num_buckets= 1000)
print("string_bucket: ", sess.run(string_bucket))


sess.close()
