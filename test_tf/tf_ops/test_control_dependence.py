import tensorflow as tf

def f1():
    x = tf.get_variable("x", shape=(), initializer=tf.zeros_initializer())
    assign_x = tf.assign(x, 10.0)
    z = x + 1.0

    with tf.train.MonitoredSession() as session:
        print(session.run(z)) #输出为1, tf只会运行最小的op集合,无关的全都不会计算


def f2():
    def func(index:int):
        x = tf.get_variable("x_"+str(index), shape=(), initializer=tf.zeros_initializer())
        assign_x = tf.assign(x, 10.0)
        z = x + 1.0
        with tf.train.MonitoredSession() as session:
            """
            注意:此处输出值无法确定,(10.0, 1.0) 或者 (10.0, 11.0). 
            The ops assign_x and z are racing(竞争)!
            """
            print(session.run([assign_x, z]))

    for i in range(20):
        func(i)


def f3():
    x = tf.get_variable("x", shape=(), initializer=tf.zeros_initializer())
    assign_x = tf.assign(x, 10.0)

    with tf.control_dependencies([assign_x]): # 利用control_dependencies控制竞争:先算完assign_x,再计算z
        z = x + 1.0
    with tf.train.MonitoredSession() as session:
        print(session.run(z)) # 11.0

def test_cond():
    v1 = tf.get_variable("v1", shape=(), initializer=tf.zeros_initializer())
    v2 = tf.get_variable("v2", shape=(), initializer=tf.zeros_initializer())
    switch = tf.placeholder(tf.bool)
    # cond依赖的内部op并不会被先执行,而是到了switch时才执行
    cond = tf.cond(switch,
                   lambda: tf.assign(v1, 1.0),
                   lambda: tf.assign(v2, 2.0))
    with tf.train.MonitoredSession() as session:
        session.run(cond, feed_dict={switch: False})
        print(session.run([v1, v2]))  # Output: (0.0, 2.0)

def test_cond2():
    v1 = tf.get_variable("v1", shape=(), initializer=tf.zeros_initializer())
    v2 = tf.get_variable("v2", shape=(), initializer=tf.zeros_initializer())
    switch = tf.placeholder(tf.bool)
    assign_v1 = tf.assign(v1, 1.0)
    assign_v2 = tf.assign(v2, 2.0)
    # cond依赖的所有外部op都会被先执行(无论true or false),因此v2会被赋值2
    cond = tf.cond(switch,
                   lambda: assign_v1,
                   lambda: assign_v2)
    with tf.train.MonitoredSession() as session:
        session.run(cond, feed_dict={switch: False})
        # 课件中认为输出为:Output: (1.0, 2.0)
        print(session.run([v1, v2]))

def test_while():
    k_limit = tf.constant(5)
    matrix = tf.ones(shape=[2, 2]) # matrix:[[1,1], [1,1]]
    condition = lambda i, _: i < k_limit # i is a tensor here, and < is thus tf.less
    body = lambda i, m: (i + 1, tf.matmul(m, matrix))
    final_i, power = tf.while_loop(
        cond=condition,
        body=body,
        loop_vars=(0, tf.diag([1., 1.]))) # [[1,0],
                                          #  [0, 1]]

    with tf.train.MonitoredSession() as session:
        print(session.run([final_i, power])) # power will be the k-th power of matrix.
        """
        [array([[16., 16.],
        [16., 16.]], dtype=float32)]  
        """

#f1()
#f2()
#f3()

#test_cond()
test_cond2()
#test_while()

