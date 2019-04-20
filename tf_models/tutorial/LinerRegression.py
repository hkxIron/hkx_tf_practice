import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.platform import tf_logging
def set_logger_file(logfile="log.txt"):
    # 此处为tf.logging添加文件日志
    from logging.handlers import RotatingFileHandler
    formatter = tf.logging._logging.Formatter('%(asctime)s %(filename)s:%(lineno)s [%(levelname)s] %(message)s')
    file_handler = RotatingFileHandler(logfile, maxBytes=50 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    tf.logging._logger.addHandler(file_handler)
    tf_logging.set_verbosity(tf.logging.INFO)

set_logger_file("log.txt")

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X*w - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

tf.logging.info("begin to train")

# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(10):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b],feed_dict={X: x,Y: y})
        tf.logging.info("Epoch: {}, w: {}, b: {}".format(epoch, w_value, b_value))
        epoch += 1

tf.logging.info("Train end!")
#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,train_X.dot(w_value)+b_value)
plt.show()