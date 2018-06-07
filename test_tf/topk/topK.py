import tensorflow as tf
import numpy as np
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()

def f1():
    print("version:",tf.VERSION)
    score = tf.constant([0.1, 0.7, 0.2, 0.6, 0.3, 0.8])
    sess = tf.Session()
    # NOTE: 如果用boolean_mask 下标就对应不上了
    # threshold = 0.3
    # mask = tf.greater(score, 0.3)
    # print("greater_mask = ", mask)
    # score = tf.boolean_mask(score, mask)
    # print("after boolean mask = ", score)

    values, indices = tf.nn.top_k(score, k=5)
    print("top_k values = ", sess.run(values))
    print("top_k indices = ", sess.run(indices))

    mask = tf.greater(values, 0.3)
    print("greater_mask = ", sess.run(mask))
    values = tf.boolean_mask(values, mask)
    indices = tf.boolean_mask(indices, mask)
    print("after mask values = ", sess.run(values))
    print("after mask indices = ", sess.run(indices))

    bags = tf.constant([1001, 1002, 1003, 1004, 1005, 1006])
    selected_bags = tf.nn.embedding_lookup(bags, indices)
    print(sess.run(selected_bags))

def f_1d():
    print("version:",tf.VERSION)
    #                        0    1    2    3    4    5
    score =     tf.constant([0.1, 0.7, 0.2, 0.6, 0.3, 0.8])
    threshold = tf.constant([0.5, 0.6, 0.2, 0.8, 0.2, 0.9])
    bags = tf.constant([1001, 1002, 1003, 1004, 1005, 1006])
    contain_seed = tf.constant([0, 1, 0, 0, 1, 1])
    random_select_num = 2
    sess = tf.Session()

    bag_len = tf.shape(score)[0]
    indices = tf.range(bag_len)
    mask = tf.greater_equal(score-threshold,0)

    print("greater_mask = ", sess.run(mask))
    score_after = tf.boolean_mask(score, mask)
    valid_indices = tf.boolean_mask(indices, mask)

    print("after mask score = ", sess.run(score_after))
    print("after valid indices = ", sess.run(valid_indices))

    selected_bags = tf.nn.embedding_lookup(bags, valid_indices)
    print("selected bags:",sess.run(selected_bags))

    selected_contain_seed = tf.nn.embedding_lookup(contain_seed, valid_indices)
    print("selected seed:",sess.run(selected_contain_seed))

    # 如果将bag与contain flag组合在一起，然后random，那么可以保持二者一一对应
    bag_contain_flag = tf.stack(values = [bags,contain_seed], axis=1)
    print("bag contain flag:", tf.shape(bag_contain_flag))
    print("bag contain flag:", sess.run(bag_contain_flag))
    tf.set_random_seed(0)
    random_bag_contain = tf.random_shuffle(bag_contain_flag,seed=0)
    print("random_bag_contain:",sess.run(random_bag_contain))
    rand_select_bag_contain = random_bag_contain[:random_select_num,:]
    r_bag = rand_select_bag_contain[:,0]
    r_contain = rand_select_bag_contain[:,1]
    print("rand_select_bag_contain:",sess.run(rand_select_bag_contain))
    print("rand_select_bag:",sess.run(r_bag))
    print("rand_select_contain:",sess.run(r_contain))
    print("res:",sess.run([rand_select_bag_contain,r_bag,r_contain]))

    # random bags
    random_bags = tf.random_shuffle(selected_bags)[:random_select_num]
    print("random bags:",sess.run(random_bags))


def f_2d():
    print("version:",tf.VERSION)
    batch =3
    bag_len = 4
    score =     tf.constant([[0.1, 0.7,0.2,0.1],
                             [0.2, 0.6,0.4,0.4],
                             [0.3, 0.8,0.2,0.4]])
    threshold = tf.constant([[0.5, 0.6,0.3,0.2],
                             [0.2, 0.8,0.2,0.3],
                             [0.2, 0.9,0.1,0.3]])
    bags = tf.constant([-1, 1001, 1002, 1003, 1004])
    sess = tf.Session()

    indices = tf.tile(tf.expand_dims(tf.range(bag_len)+1,axis=0),[batch,1]) #1,4
    mask = tf.greater_equal(score-threshold,0)

    print("greater_mask = ", sess.run(mask))
    score_after = tf.boolean_mask(score, mask)
    valid_indices = tf.boolean_mask(indices, mask)

    print("after mask score = ", sess.run(score_after))
    print("after valid indices = ", sess.run(valid_indices))

    selected_bags = tf.nn.embedding_lookup(bags, valid_indices)
    print(sess.run(selected_bags))


    # gather:
    masked_indices= indices*tf.cast(mask,dtype=tf.int32)
    #masked_indices= indices*tf.cond(mask,true_fn=lambda x:1,false_fn=lambda x:0)
    gathered_bags = tf.gather(bags,masked_indices)
    print("gathered bags:",sess.run(gathered_bags))

    # random bags
    random_select_num = 2
    #random_bags = tf.random_shuffle(selected_bags)[:random_select_num]
    #print(sess.run(random_bags))

def f_1d_new():
    print("version:",tf.VERSION)
    random_select_num = 2
    sess = tf.Session()
    #                        0    1    2    3    4    5
    score =     tf.constant([0.1, 0.7, 0.2, 0.6, 0.3, 0.8])
    threshold = tf.constant([0.5, 0.6, 0.2, 0.8, 0.2, 0.9])
    bags = tf.constant([1001, 1002, 1003, 1004, 1005, 1006])
    contain_seed = tf.constant([0, 1, 0, 0, 1, 1])
    bag_contain_flag = tf.stack(values = [bags,contain_seed], axis=1) # 第一列是bag_id, 第二列是 contain_seed
    print("bag_contain_flag:\n", sess.run(bag_contain_flag))

    bag_len = tf.shape(score)[0]
    indices = tf.range(bag_len)
    mask = tf.greater_equal(score-threshold,0)

    print("greater_mask = ", sess.run(mask))
    score_after = tf.boolean_mask(score, mask)
    valid_indices = tf.boolean_mask(indices, mask)

    print("after mask score = ", sess.run(score_after))
    print("after valid indices = ", sess.run(valid_indices))

    selected_bag_contain_seed = tf.nn.embedding_lookup(bag_contain_flag, valid_indices)
    print("selected bag_index_contain_seed:",sess.run(selected_bag_contain_seed))

    # 如果将bag与contain flag组合在一起，然后random，那么可以保持二者一一对应
    random_bag_contain = sess.run(tf.random_shuffle(bag_contain_flag,seed=0))
    print("random_bag_contain:",random_bag_contain)
    rand_select_bag_contain = random_bag_contain[:random_select_num,:]
    r_bag = rand_select_bag_contain[:,0]
    r_contain = rand_select_bag_contain[:,1]
    print("rand_select_bag_contain:",rand_select_bag_contain)
    print("rand_select_bag:",r_bag)
    print("rand_select_contain:",r_contain)

def get_bits(num_list):
    arr=0x0
    mask=2**64-1
    for x in num_list:
        arr|=(mask&(1<<x))
    return arr&mask

def f_1d_site_set():
    print("version:",tf.VERSION)
    input_site_set=25
    random_select_num = 2
    sess = tf.Session()
    #                        0    1    2    3    4    5
    score =     tf.constant([0.1, 0.7, 0.2, 0.6, 0.3, 0.8])
    threshold = tf.constant([0.5, 0.6, 0.2, 0.8, 0.2, 0.9])
    bags = tf.constant([1001, 1002, 1003, 1004, 1005, 1006])
    #分站点，每个号码包包含的站点列表
    bag_site_set_list = [
        [15,21,25,28],
        [15,21,25],
        [21,25,28],
        [21,25],
        [],
        [25]
    ]
    bag_set_list_bits=[get_bits(x) for x in bag_site_set_list]
    # 测试bit测试是否有效
    print("test result:",bag_set_list_bits[0]&get_bits([25,21,15])>0)
    print("test result:",bag_set_list_bits[0]&get_bits([25,21,14])==0)
    contain_list = [ x&get_bits([input_site_set])>0 for x in bag_set_list_bits]
    print("contain list:", contain_list)

    contain_list_tensor = tf.constant(contain_list,dtype=tf.bool)
    contain_seed = tf.constant([0, 1, 0, 0, 1, 1])
    bag_contain_flag = tf.stack(values = [bags,contain_seed], axis=1) # 第一列是bag_id, 第二列是 contain_seed
    print("bag_contain_flag:\n", sess.run(bag_contain_flag))

    bag_len = tf.shape(score)[0]
    indices = tf.range(bag_len)
    threshold_mask = tf.greater_equal(score-threshold,0)
    mask = tf.logical_and(threshold_mask, contain_list_tensor)
    print("threshold_mask:",sess.run(threshold_mask), "\ncontain_list_tensor:",sess.run(contain_list_tensor), "\nmask:",sess.run(mask))

    print("greater_mask = ", sess.run(mask))
    score_after = tf.boolean_mask(score, mask)
    valid_indices = tf.boolean_mask(indices, mask)

    print("after mask score = ", sess.run(score_after))
    print("after valid indices = ", sess.run(valid_indices))

    selected_bag_contain_seed = tf.nn.embedding_lookup(bag_contain_flag, valid_indices)
    print("selected bag_index_contain_seed:",sess.run(selected_bag_contain_seed))

    # 如果将bag与contain flag组合在一起，然后random，那么可以保持二者一一对应
    random_bag_contain = sess.run(tf.random_shuffle(selected_bag_contain_seed,seed=0))
    print("random_bag_contain:",random_bag_contain)
    rand_select_bag_contain = random_bag_contain[:random_select_num,:]
    r_bag = rand_select_bag_contain[:,0]
    r_contain = rand_select_bag_contain[:,1]
    print("rand_select_bag_contain:",rand_select_bag_contain)
    print("rand_select_bag:", r_bag)
    print("rand_select_contain:", r_contain)

#f_1d_new()
#f_1d()
f_1d_site_set()
