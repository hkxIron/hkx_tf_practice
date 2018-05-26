import tensorflow as tf

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
    print(sess.run([value]))
    print(sess.run([result]))


def test_t2():
    sp1 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 2, 1], dense_shape=(2, 3))  #
    sp2 = tf.SparseTensor(indices=[[0, 0], [1, 0]], values=[-1,-2], dense_shape=(2, 2))  #

    sp_con = tf.sparse_concat(axis=1,sp_inputs=[sp1,sp2])
    index_ones = tf.ones(shape=tf.shape(sp1.values), dtype=tf.float32)
    index_feat_weights = tf.SparseTensor(indices=sp1.indices, values=index_ones,
                                         dense_shape=sp1.dense_shape)

    sess = tf.Session()
    print("sp1",sess.run([sp1]))
    print("sp_con",sess.run([sp_con]))
    print("index_ones",sess.run([index_ones]))
    print("index_feat_weights",sess.run([index_feat_weights]))

def test_t3():
    print("----------------------")
    sp1 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[1, 123, 1], dense_shape=(2, 3))  #
    sp1_dense = tf.sparse_tensor_to_dense(sp1,default_value=-1)
    indicate= tf.sparse_to_indicator(sp1,vocab_size=123+1)
    sess = tf.Session()
    print("sp1:",sess.run(sp1))
    print("sp_dense:",sess.run(sp1_dense))
    print("indicate:",sess.run(indicate))

def count_sparse_nonzero(sp,axis,add_default=0):
    sp_non_negative = tf.SparseTensor(indices=sp.indices, values=sp.values + add_default, dense_shape=sp.dense_shape)
    return tf.sparse_reduce_sum(tf.sign(sp_non_negative),axis)

def test_t4():
    print("----------------------")
    sp1 = tf.SparseTensor(indices=[[0, 0], [0, 1], [1, 0]], values=[12, -1, 0], dense_shape=(3, 3))  #
    #sp1.values = sp1.values+1
    sp1_dense = tf.sparse_tensor_to_dense(sp1,default_value=-1)
    sess = tf.Session()
    print("sp1:",sess.run(sp1))
    sp1_sign = tf.sign(sp1)
    sp2 = tf.SparseTensor(indices=sp1.indices, values=sp1.values+2, dense_shape=sp1.dense_shape)  #
    #sp1_sign = tf.sign(tf.sparse_add(sp1)))
    #print("sp1_sign:",sess.run(sp1_sign))
    print("sp2:",sess.run(sp2))
    sp2_sign = tf.sign(sp2)
    print("sp2_sign:",sess.run(sp2_sign))
    print("sp2_sign_sum:",sess.run(tf.sparse_reduce_sum(sp2_sign,axis=1)))
    print("sp2_sign_sum_fun:",sess.run(count_sparse_nonzero(sp1,axis=1,add_default=2)))

def test_t5():
    sess =tf.Session()
    sp1= tf.constant(value=1.0,dtype=tf.float32)
    print(sess.run(sp1))

    sp2= tf.cast(sp1,dtype=tf.int8)
    print(sess.run(sp2))

def main():
    #test_t1()
    #test_t2()
    #test_t3()
    #test_t4()
    test_t5()

if __name__ == '__main__':
    tf.app.run(main=main)