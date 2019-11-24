import numpy as np
import tensorflow as tf
#import matplotlib
import matplotlib.pyplot as plt

num_samples, w, b = 20, 0.5, 2.
xs = np.asarray(range(num_samples))
ys = np.asarray([
x * w + b + np.random.normal()
for x in range(num_samples)])
plt.plot(xs, ys)
plt.show()

class Linear(object):
    def __init__(self):
        self.w = tf.get_variable(
        "w",dtype=tf.float32,shape=[],initializer=tf.zeros_initializer())
        self.b = tf.get_variable(
        "b",dtype=tf.float32,shape=[],initializer=tf.zeros_initializer())
    def __call__(self, x):
        return self.w * x + self.b

xtf = tf.placeholder(tf.float32, [num_samples], "xs")
ytf = tf.placeholder(tf.float32, [num_samples], "ys")
model = Linear()
model_output = model(xtf) #调用__call__函数

cov = tf.reduce_sum((xtf-tf.reduce_mean(xtf))*(ytf-tf.reduce_mean(ytf)))
var = tf.reduce_sum(tf.square(xtf-tf.reduce_mean(xtf)))
w_hat = cov / var
b_hat = tf.reduce_mean(ytf)-w_hat*tf.reduce_mean(xtf)

# 通过assign手动更新梯度
solve_w = model.w.assign(w_hat)
solve_b = model.b.assign(tf.reduce_mean(ytf)-w_hat*tf.reduce_mean(xtf))

with tf.train.MonitoredSession() as sess:
    sess.run([solve_w, solve_b], feed_dict={xtf: xs, ytf: ys})
    preds = sess.run(model_output, feed_dict={xtf: xs, ytf: ys})

plt.plot(xs, ys)
plt.plot(xs, preds)
plt.show()


# 方法二:利用sgd进行迭代
loss = tf.losses.mean_squared_error(ytf, model_output)
grads = tf.gradients(loss, [model.w, model.b])
update_w = tf.assign(model.w, model.w - 0.001 * grads[0])
update_b = tf.assign(model.b, model.b - 0.001 * grads[1])
update = tf.group(update_w, update_b)

plt.plot(xs, ys)
feed_dict = {xtf: xs, ytf: ys}
with tf.train.MonitoredSession() as sess:
    for i in range(500):
        sess.run(update, feed_dict=feed_dict)
        if i in [1, 5, 25, 125, 499]:
            preds = sess.run(model_output, feed_dict=feed_dict)
            plt.plot(xs, preds, label=str(i))

#plt.legend("不同的迭代次数拟合直线")
plt.legend("")
plt.show()

