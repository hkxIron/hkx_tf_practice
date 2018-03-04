# coding: utf-8

def t1():
    import tensorflow as tf
    # 注意， bias1 的定义方式
    with tf.variable_scope('v_scope') as scope1:
        Weights1 = tf.get_variable('Weights', shape=[2,3])
        bias1 = tf.Variable(initial_value=[0.52], name='bias')

    print("Weights1:",Weights1.name) # Weights1: v_scope/Weights:0
    print("bias1:",bias1.name) # bias1: v_scope/bias:0

    # 下面来共享上面已经定义好的变量
    # note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
    with tf.variable_scope('v_scope', reuse=True) as scope2:
        Weights2 = tf.get_variable('Weights')
        print("Weights2:",Weights2.name) # Weights2: v_scope/Weights:0
        bias2 = tf.get_variable('bias')
        print("bias2:",bias2.name) # 报错,tf.get_variable获取的变量必须由 tf.get_varibale函数创建,而不能是 tf.Variable

def t2():
    import tensorflow as tf
    # 注意， bias1 的定义方式
    with tf.variable_scope('v_scope') as scope1:
        Weights1 = tf.get_variable('Weights', shape=[2, 3])
    #    bias1 = tf.Variable([0.52], name='bias')

    # 下面来共享上面已经定义好的变量
    # note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
    with tf.variable_scope('v_scope', reuse=True) as scope2:
        Weights2 = tf.get_variable('Weights')
        bias2 = tf.Variable([0.52], name='bias')

    print(Weights1.name) # v_scope/Weights:0
    print(Weights2.name) # v_scope/Weights:0
    print(bias2.name)    # v_scope_1/bias:0

t1()
#t2()