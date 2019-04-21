import tensorflow as tf
import numpy as np

N, D, H = 64, 10, 100
learning_rate = 1e-4
iter = 500
print_iter = 100

def weights_on_fed():
    print("weights on fed")
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))
    w1 = tf.placeholder(tf.float32, shape=(D, H))
    w2 = tf.placeholder(tf.float32, shape=(H, D))

    h = tf.maximum(tf.matmul(x, w1), 0)
    y_pred = tf.matmul(h, w2)

    diff = y_pred - y
    loss = tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))

    # 计算梯度
    grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

    with tf.Session() as sess:
        values = {x: np.random.randn(N, D),
                  y:np.random.randn(N, D),
                  w1:np.random.randn(D, H),
                  w2:np.random.randn(H, D),
                  }

        for t in range(iter):
            out = sess.run([loss, grad_w1, grad_w2],
                            feed_dict=values
                           )
            loss_val, grad_w1_val, grad_w2_val = out
            if t%print_iter==0:
                print("iter:{} loss:{}".format(t, loss_val))
            # 手动更新w, 第一次见
            """
            problem: copying weights between CPU / GPU each step
            """
            values[w1] -= learning_rate * grad_w1_val
            values[w2] -= learning_rate * grad_w2_val
weights_on_fed()


"""
Change w1 and w2 from placeholder (fed on each call) to Variable
(persists in the graph between calls)
"""
def weights_persist_in_graph():
    print("weights persist in graph")
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))
    w1 = tf.Variable(tf.random_normal((D, H)))
    w2 = tf.Variable(tf.random_normal((H, D)))

    h = tf.maximum(tf.matmul(x, w1), 0)
    y_pred = tf.matmul(h, w2)

    diff = y_pred - y
    loss = tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))

    # 计算梯度
    grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

    #
    new_w1 = w1.assign(w1 - learning_rate* grad_w1)
    new_w2 = w2.assign(w2 - learning_rate* grad_w2)

    """
    Add dummy graph node that depends on updates
    """
    updates = tf.group(new_w1, new_w2)

    with tf.Session() as sess:
        """
        Run graph once to initialize w1 and w2
        """
        sess.run(tf.global_variables_initializer())
        values = {x: np.random.randn(N, D),
                  y:np.random.randn(N, D),
                  }
        """
        Run many times to train
        """
        for t in range(iter):
            loss_val, _ = sess.run([loss, updates], feed_dict=values)
            if t%print_iter==0:
                print("iter:{} loss:{}".format(t, loss_val))

weights_persist_in_graph()

def weights_optimizer_graph():
    print("weights optimizer in graph")
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))
    w1 = tf.Variable(tf.random_normal((D, H)))
    w2 = tf.Variable(tf.random_normal((H, D)))

    h = tf.maximum(tf.matmul(x, w1), 0)
    y_pred = tf.matmul(h, w2)

    diff = y_pred - y
    # loss = tf.reduce_mean(tf.reduce_sum(diff**2, axis=1))
    loss = tf.losses.mean_squared_error(y_pred, y)


    # 计算梯度
    # grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    updates = optimizer.minimize(loss)

    with tf.Session() as sess:
        """
        Run graph once to initialize w1 and w2
        """
        sess.run(tf.global_variables_initializer())
        values = {x: np.random.randn(N, D),
                  y:np.random.randn(N, D),
                  }
        for t in range(iter):
            loss_val, _ = sess.run([loss, updates], feed_dict=values)
            if t%print_iter==0:
                print("iter:{} loss:{}".format(t, loss_val))
weights_optimizer_graph()

def tf_layer_test():
    print("tf layer test")
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))

    # Use He initializer
    init = tf.variance_scaling_initializer(2.0)
    h = tf.layers.dense(inputs=x,
                        units=H,
                        activation=tf.nn.relu,
                        kernel_initializer=init
                        )
    y_pred = tf.layers.dense(
       inputs=h,
        units=D,
        kernel_initializer=init
    )

    loss = tf.losses.mean_squared_error(y_pred, y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    updates = optimizer.minimize(loss)

    with tf.Session() as sess:
        """
        Run graph once to initialize w1 and w2
        """
        sess.run(tf.global_variables_initializer())
        values = {x: np.random.randn(N, D),
                  y:np.random.randn(N, D),
                  }
        for t in range(iter):
            loss_val, _ = sess.run([loss, updates], feed_dict=values)
            if t%print_iter==0:
                print("iter:{} loss:{}".format(t, loss_val))

tf_layer_test()

def tf_keras():
    print("tf keras test")
    x = tf.placeholder(tf.float32, shape=(N, D))
    y = tf.placeholder(tf.float32, shape=(N, D))

    # Use He initializer
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(H, input_shape=(D, ),
                                    activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(D))
    y_pred = model(x)

    loss = tf.losses.mean_squared_error(y_pred, y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    updates = optimizer.minimize(loss)

    with tf.Session() as sess:
        """
        Run graph once to initialize w1 and w2
        """
        sess.run(tf.global_variables_initializer())
        values = {x: np.random.randn(N, D),
                  y:np.random.randn(N, D),
                  }
        for t in range(iter):
            loss_val, _ = sess.run([loss, updates], feed_dict=values)
            if t%print_iter==0:
                print("iter:{} loss:{}".format(t, loss_val))
tf_keras()

def tf_keras2():
    print("tf keras2 test")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(H, input_shape=(D, ),
                                    activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(D))

    model.compile(loss= tf.keras.losses.mean_squared_error,
                  optimizer=tf.keras.optimizers.SGD(lr=learning_rate)
                  )
    x =np.random.randn(N, D)
    y =np.random.randn(N, D)
    """
    Keras can handle the training loop for you! No sessions or feed_dict
    """
    history = model.fit(x, y, epochs=50, batch_size=N)
    # print("history", history)

tf_keras2()
