import tensorflow as tf

# tf.control_dependencies不work
def f1():
    print("f1")
    w = tf.Variable(1.0)
    ema = tf.train.ExponentialMovingAverage(0.9)
    update = tf.assign_add(w, 1.0)

    ema_op = ema.apply([update]) # update all shadow variables
    with tf.control_dependencies([ema_op]):
        ema_val = ema.average(update)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(3):
            print(sess.run([ema_val]))


# tf.control_dependencies work
def f2():
    print("f2")
    w = tf.Variable(1.0)
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    update = tf.assign_add(w, 1.0)

    ema_op = ema.apply([update])
    with tf.control_dependencies([ema_op]):
        ema_val = tf.identity(ema.average(update))  # 一个identity搞定

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(3):
            print(sess.run([ema_val]))

f1()
f2()
