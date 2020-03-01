import tensorflow as tf

"""
  If `name_or_scope` is not None, it is used as is. If `name_or_scope` is None,
  then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will be made unique by appending `_N`
  to it.

  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](https://tensorflow.org/guide/variables), here
  we present only a few basic examples.

  Simple example of how to create a new variable:
"""

def f1():
    with tf.variable_scope("foo"):
        with tf.variable_scope("bar"):
            v = tf.get_variable("v", [1])
            print("v.name", v.name)
            assert v.name == "foo/bar/v:0"

"""
  Simple example of how to reenter a premade variable scope safely:
"""
def f2():
    with tf.variable_scope("foo") as vs:
        print("variable scope:foo")
        pass

    # Re-enter the variable scope.
    with tf.variable_scope(vs,  # 这里是利用上面的foo空间
                           auxiliary_name_scope=False) as vs1:
        # Restore the original name_scope.
        with tf.name_scope(vs1.original_name_scope):
            v = tf.get_variable("v", [1])
            assert v.name == "foo/v:0"
            c = tf.constant([1], name="c")
            assert c.name == "foo/c:0"

"""
Basic example of sharing a variable AUTO_REUSE:
"""
def f3():
    def foo():
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE): # 自动重用
            v = tf.get_variable("v", [1])
        return v

    v1 = foo()  # Creates v.
    v2 = foo()  # Gets the same, existing v.
    assert v1 == v2

"""
 Basic example of sharing a variable with reuse=True:
"""
def f4():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
    # 重用上面的空间, 变量是一致的
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v", [1])
    assert v1 == v

"""
  Sharing a variable by capturing a scope and setting reuse:
"""
def f5():
    with tf.variable_scope("foo") as scope:
        v = tf.get_variable("v", [1])
        print("v name:", v.name) # foo/v:0
        v2 = tf.get_variable("v2", [1])
        print("v2 name:", v2.name) # foo/v2:0
        assert v2 != v
        scope.reuse_variables() # 手动捕获当前的scope,然后
        v1 = tf.get_variable("v", [1])
        print("v1 name:", v1.name)
    assert v1 == v # foo/v:0

"""
  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.
"""
def f6():
  with tf.variable_scope("foo"):
      v = tf.get_variable("v", [1])
      v1 = tf.get_variable("v", [1]) # 变量已经存在, 但未说明要生用
      #  Raises ValueError("... v already exists ...").
      """
      ValueError: Variable foo/v already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
      """


"""
  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.
"""
def f7():
  # 如果变量不存在,而声明reuse会出错
  with tf.variable_scope("foo", reuse=True):
      v = tf.get_variable("v", [1]) # 即变量还未被创建,就声明重用,所以找不到
      #  Raises ValueError("... v does not exists ...").
      # ValueError: Variable foo/v does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=tf.AUTO_REUSE in VarScope?

def f8():
    # 如果变量不存在,而声明 auto_reuse 不会出错
    with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        v = tf.get_variable("v", [1])
f1()
#f2()
#f3()
#f4()
#f5()
#f6()
#f7()
#f8()