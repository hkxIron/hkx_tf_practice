import tensorflow as tf
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

f1()
