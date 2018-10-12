# coding: utf-8
import tensorflow as tf
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

def t3():
    import tensorflow as tf
    import numpy as np
    # 注意， bias1 的定义方式
    np.random.seed(0)
    weight = np.random.random((3,2)).transpose()
    sess =tf.Session()
    print("weight:",weight)
    with tf.variable_scope('v_scope') as scope1:
        weight_var = tf.get_variable('Weights',shape=[2,3])
        print("var1:",weight_var.name) # v_scope/Weights:0

        scope1.reuse_variables() # 如果不加则下面get_variable会报错
        weight_var2 = tf.get_variable('Weights') # v_scope/Weights
        print("var2:",weight_var2.name) # v_scope/Weights:0

        #weight_var3 = tf.get_variable('v_scope/Weights') # v_scope/v_scope/Weights,这样获取是错的
        #print(weight_var3.name) # 报错：ValueError: Variable v_scope/v_scope/Weights does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?

        #tf.transpose(weight_var)
    #bias1 = tf.Variable([0.52], name='bias')
    weight_var4 = tf.get_variable('Weights', shape=[2,3])  # Weights:0, 注意，此处创建了新变量，var值跟原来不一样!!!不是同一个变量。
    print("var4:",weight_var4.name) # Weights:0

    tf.get_variable_scope().reuse_variables() # 设置变量重用
    weight_var5 = tf.get_variable('v_scope/Weights')  # v_scope/Weights:0,
    print("var5:",weight_var5.name) # v_scope/Weights:0, 获取的是var1的值

    init = tf.global_variables_initializer()
    sess.run(init)
    print("weight:",sess.run(weight_var))
    sess.close()

def test_var2():
    import tensorflow as tf
    import numpy as np
    a = tf.Variable(np.arange(8).reshape(2, 2, 2))
    b = tf.Variable(np.arange(8))
    print("before reshape:", tf.shape(b))
    b = tf.reshape(b, tf.shape(a))
    print("before reshape:", tf.shape(a))
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print("a", sess.run([a]))
    print("b", sess.run([b]))

def t4():
    import numpy as np
    a = np.arange(8).reshape(2, 2, 2)
    b = np.arange(8).reshape(2, 2, 2)
    res= np.equal(a,b)
    print(np.alltrue(res))
    np.savetxt()

def test_abs_name_scope():
    with tf.Graph().as_default() as g:
        with g.name_scope("nested") as scope:
            nested_c = tf.constant(10.0, name="c")
            print("nested_c: ", nested_c.op.name)

            assert nested_c.op.name == "nested/c"
            # Create a nested scope called "inner".
            with g.name_scope("inner"):
                nested_inner_c = tf.constant(30.0, name="c")
                print("nested_inner_c name: ", nested_inner_c)
                assert nested_inner_c.op.name == "nested/inner/c"

                # Treats `scope` as an absolute name scope,
                # and switches to the "nested/" scope.
                with g.name_scope(scope):
                    nested_d = tf.constant(40.0, name="d")
                    assert nested_d.op.name == "nested/d"

                    # reset name scope
                    with g.name_scope(""):
                        e = tf.constant(50.0, name="e")
                    assert e.op.name == "e"
#t2()
t3()
#t4()

#test_abs_name_scope()