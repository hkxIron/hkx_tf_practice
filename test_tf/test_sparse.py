from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def test_embedding_lookup():
    a = np.arange(8).reshape(2,4)
    b = np.arange(8,12).reshape(1,4)
    c = np.arange(12, 20).reshape(2,4)
    print("a:",a)
    print("b:",b)
    print("c:",c)

    a = tf.Variable(a)
    b = tf.Variable(b)
    c = tf.Variable(c)

    t1 = tf.nn.embedding_lookup([a], ids=[0,1])
    t2 = tf.nn.embedding_lookup([a,b], ids=[0,1])
    t3 = tf.nn.embedding_lookup([a,b,c], ids=[0,1])
    t4 = tf.nn.embedding_lookup([a,b,c], ids=[0,1,2])
    t5 = tf.nn.embedding_lookup([a,b,c], ids=[0,1,2,3])

    # 此处如果ids=[0,1,2,3]不会报错，因为此时并没有发现b比c要少一行，程序能够正常的执行，但是如果出现参数4了，因为
    # 程序的partition要求在无法进行均匀切分时，前面的(max_id+1)%len(params)个param的切分可以多一个。在此例子中
    # 正确的id应该是params中的第一元素的id为[0,3], 第二元素的id应该为[1,4], 第三个元素的id应该为[2]。所以正确的param
    # 应该是（a,c,b)或者(c,a,b),总之b应该放在最后面
    # 本例的运算结果为：
    '''
    t = tf.nn.embedding_lookup([a,b,c], ids=[0,1,2,3])
 
    [[ 0  1  2  3]
    [ 8  9 10 11]
    [12 13 14 15]
    [ 4  5  6  7]]
    '''
    #但是本例中的[a,b,c]顺序其实是错误的

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run([t1,t2,t3,t4,t5]))

def test_lookup_sparse():
    a = np.arange(8).reshape(2, 4)
    b = np.arange(8, 16).reshape(2, 4)
    c = np.arange(12, 20).reshape(2, 4)

    print("a:",a)
    print("b:",b)
    print("c:",c)

    a = tf.Variable(a, dtype=tf.float32)
    b = tf.Variable(b, dtype=tf.float32)
    c = tf.Variable(c, dtype=tf.float32)
    """
    idx = tf.SparseTensor(indices=[[0,0], [0,2], [1,0], [1, 1]], values=[1,2,2,0], dense_shape=(2,3))
    result = tf.nn.embedding_lookup_sparse((a,c,b), idx, None, combiner="sum")
    """
    """
a :[[0 1 2 3]
    [4 5 6 7]]
b:[[ 8  9 10 11]
   [12 13 14 15]]
c:[[12 13 14 15]
   [16 17 18 19]]
"""
    # values:是指embedding里的index_id, 不是指权重
    idx = tf.SparseTensor(indices=[[0,0],[0,1],[1,0],[1,1]], values=[0,1,0,0], dense_shape=(2,3)) #
    combine = [a,b]
    result = tf.nn.embedding_lookup_sparse(combine, idx, None, combiner="sum")

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    idx_res,result,combine = sess.run([idx,result,combine])
    print("combine:",combine)
    print("idx_res:",idx_res)
    print("result:",result)
'''
根据程序的测试结果来看，这里的params的结合方式并不是成为一个逻辑大tensor，而是直接变成一个大的tensor，在该tensor的在第0维扩张
idx = tf.SparseTensor(indices=[[0,0], [0,2], [1,0], [1, 1]], values=[1,2,2,0], dense_shape=(2,3))
result = tf.nn.embedding_lookup_sparse((a,b,c), idx, None, combiner="sum")

a,b,c 为：
a :[[0 1 2 3]
 [4 5 6 7]]

b:[[ 8  9 10 11]
 [12 13 14 15]]

c:[[12 13 14 15]
 [16 17 18 19]]

 在实现中好像将它们结合成一个大的tensor了，而不是使用partition,即实现的结果是
[[[0 1 2 3]
 [4 5 6 7]]
[[ 8  9 10 11]
 [12 13 14 15]]
[[12 13 14 15]
 [16 17 18 19]]
]
 最后的结果为：
 [[[ 20.  22.  24.  26.]
  [ 28.  30.  32.  34.]]

 [[  8.  10.  12.  14.]
  [ 16.  18.  20.  22.]]]

    '''

def test_all():
    feat_num = 5
    feat_dim = 2
    tf.set_random_seed(0)
    feat_embedding_w = tf.Variable(tf.random_uniform([feat_num,feat_dim],-1.0,1.0))
    sp_ids= tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[0, 1, 0, 0], dense_shape=(2, 3))  #
    feature_embed = tf.nn.embedding_lookup_sparse(feat_embedding_w,
                                                  sp_ids=sp_ids,
                                                  sp_weights=None,
                                                  combiner="sum"
                                                  )
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    feat_embedding_w, sp_ids, feature_embed=sess.run([feat_embedding_w,sp_ids,feature_embed])
    print("feat_embedding_w:",feat_embedding_w,
          "\nsp_ids:",sp_ids,
          "\nfeature_embed:",feature_embed)
    sess.close()


def test_idx():
    feat_num = 5
    feat_dim = 2
    tf.set_random_seed(0)
    feat_embedding_w = tf.Variable(tf.random_uniform([feat_num,feat_dim],-1.0,1.0))
    sp_value_ids= tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 1], dense_shape=(2, 2))  #
    dim = sp_value_ids.get_shape()[0]
    print("dim:",dim)
    #ind = np.concatenate((np.arange(dim).reshape((dim,1)),np.zeros(dim).reshape((dim,1))),axis=1)
    #sp_cons= tf.SparseTensor(indices=ind, values=tf.ones(dim), dense_shape=(dim,1))  #
    sp_cons =1
    #sp_value_ids= tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 1], dense_shape=(2, 2))  #
    sp_index_ids= tf.SparseTensor(indices=[[0, 0], [1, 0], [1, 1],[1,2]], values=[3, 4, 1, 0], dense_shape=(2, 3))  #
    sp_ids = tf.sparse_concat(1,[sp_value_ids,sp_index_ids])
    sp_value_ids_ones= sp_ids
    #print(tf.shape(sp_value_ids))
    d1=sp_value_ids.get_shape()[0]
    d2=sp_value_ids.get_shape()[1]
    #ele_num = sp_value_ids.values.get_shape()[0]
    ele_num = sp_value_ids.indices.get_shape()[0]
    print("---------shape:",ele_num)
    sp_value_ids_ones= tf.SparseTensor(indices=sp_value_ids.indices,values=np.ones(ele_num), dense_shape=sp_value_ids.dense_shape)  #
    #sp_value_ids_ones= tf.SparseTensor(indices=sp_value_ids.indices,values=sp_value_ids.values, dense_shape=sp_value_ids.dense_shape)  #
    feature_embed = tf.nn.embedding_lookup_sparse(feat_embedding_w,
                                                  sp_ids=sp_ids,
                                                  sp_weights=None,
                                                  combiner="sum"
                                                  )
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    feat_embedding_w, sp_ids, feature_embed,sp_value_ids_ones ,indices =sess.run([feat_embedding_w,sp_ids,feature_embed,sp_value_ids_ones,sp_value_ids.indices])
    print("feat_embedding_w:",feat_embedding_w,
          "\nsp_ids:",sp_ids,
          "\nfeature_embed:",feature_embed,
          "\nsp_value_ids_ones:",sp_value_ids_ones,
          "\nindices:",indices,
          "\nsp_cons:",sp_cons
          )
    sess.close()


def test_sparse_count():
    feat_num = 5
    feat_dim = 2
    tf.set_random_seed(0)
    feat_embedding_w = tf.Variable(tf.random_uniform([feat_num, feat_dim], -1.0, 1.0))
    sp_value_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 1], dense_shape=(2, 2))  #
    dense=tf.sparse_tensor_to_dense(sp_value_ids)
    sp_value_ids_count =tf.count_nonzero(dense,axis=1,keep_dims=True)
    #sp_value_ids_count =tf.reshape(tf.count_nonzero(dense,axis=1),[tf.shape(dense)[0],1])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    sp_value_ids,sp_value_ids_count = sess.run(
        [ sp_value_ids,sp_value_ids_count])
    print(
        "\nsp_value_ids", sp_value_ids,
        "\nsp_value_ids_count:", sp_value_ids_count,
          )
    sess.close()


dt ={ 0:4,1:0,2:1,3:2,4:3 }
def get_index(x):
    print("x:",x)
    return dt[x]

def get_sq(x):
    print("x:",x)
    return x*x

def test_tensor():
    feat_embedding_w = tf.constant([[3,2],[1,4]])
    #new_w = tf.map_fn(get_index,feat_embedding_w)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    #feat_embedding_w_res,new_w_res =sess.run([feat_embedding_w,new_w])
    #print(feat_embedding_w_res,new_w_res)

    elems = np.array([1, 2, 3, 4])
    #elems = np.array([[1, 2, 3],[ 4, 5, 6]])
    elems_tensor = tf.convert_to_tensor(elems)
    squares = tf.py_func(get_index,[elems_tensor],tf.int64)
    #squares = tf.map_fn(lambda x: dt[x], elems_tensor)
    #squares = tf.map_fn(lambda x: x * x, elems_tensor)
    #squares = tf.map_fn(get_sq, elems_tensor)
    squares = sess.run([squares])
    #tf.squeeze
    #tf.py_func()
    print("squares:",squares)

def is_non_neg(x):
    if x >= 0: return 1
    else: return 0

def test_t1():
    sp_value_ids = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 1], dense_shape=(2, 2))  #
    #dense = tf.sparse_to_dense()
    dense = tf.sparse_tensor_to_dense(sp_value_ids)
    zeros = tf.zeros_like(dense)
    value = tf.cast(tf.greater(dense,zeros),dtype=tf.float32)
    result = tf.reduce_sum(value,axis=1)
    #dense_non_neg = tf.map_fn(is_non_neg,dense)
    #indicater= tf.sparse_to_indicator(sp_value_ids)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    print(sess.run([sp_value_ids,value,result]))



"""
def test_sparse_map_fn():
    feat_num = 5
    feat_dim = 2
    tf.set_random_seed(0)
    feat_embedding_w = tf.Variable(tf.random_uniform([feat_num,feat_dim],-1.0,1.0))
    sp_ids= tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0], [1, 1]], values=[2, 3, 1,4], dense_shape=(2, 3))  #
    #sp_id_indexs= tf.map_fn(get_index,sp_ids)
    #tf.scatter_nd_update()
    sp_id_indexs= tf.SparseTensor(sp_ids.indices,tf.get_index(sp_ids.values),sp_ids.dense_shape)
    #sp_id_indexs= tf.SparseTensor(sp_ids.indices,tf.map_fn(get_index,sp_ids.values),sp_ids.dense_shape)
    feature_embed = tf.nn.embedding_lookup_sparse(feat_embedding_w,
                                                  sp_ids=sp_id_indexs,
                                                  sp_weights=None,
                                                  combiner="sum"
                                                  )
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    feat_embedding_w, sp_ids, feature_embed,sp_id_indexs=sess.run([feat_embedding_w,sp_ids,feature_embed,sp_id_indexs])
    print("feat_embedding_w:",feat_embedding_w,
          "\nsp_ids:",sp_ids,
          "\nfeature_embed:",feature_embed,
          "\nsp_id_indexs:",sp_id_indexs
          )
    sess.close()
"""

def main(_):
    test_embedding_lookup()
    #test_lookup_sparse()
    #test_all()
    #test_sparse_map_fn()
    #test_tensor()
    #test_idx()
    #test_sparse_count()
    #test_t1()

if __name__ == '__main__':
    tf.app.run(main=main)
