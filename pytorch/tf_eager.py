import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

print("tf version:", tf.__version__)

"""
tensorflow 动态图,动态创建图,
而不需要用tf.cond, tf.foldl等特殊的op
"""
def eager_mode_tf():

    N, D = 3, 4
    """
    These calls to tf.random_normal produce concrete values! No need
    for placeholders / sessions Wrap values in a tfe.Variable if we might
    want to compute grads for them
    """
    x = tfe.Variable(tf.random_normal((N, D)))
    y = tfe.Variable(tf.random_normal((N, D)))
    z = tfe.Variable(tf.random_normal((N, D)))

    """
    Operations scoped under a GradientTape will build a
    dynamic graph, similar to PyTorch
    """
    with tfe.GradientTape() as tape:
        a = x * z
        b= a + z
        c = tf.reduce_sum(b)

    grad_x, grad_y, grad_z = tape.gradient(c, [x, y, z])
    """
    Use the tape to compute gradients, like .backward()
    in PyTorch. The print statement works!
    """
    print("eager tf:", grad_x)

eager_mode_tf()