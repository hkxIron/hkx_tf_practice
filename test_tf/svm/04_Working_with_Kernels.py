# https://github.com/hkxIron/tensorflow_cookbook/blob/master/04_Support_Vector_Machines/04_Working_with_Kernels/04_svm_kernels.py
# Illustration of Various Kernels
# ----------------------------------
#
# This function wll illustrate how to
# implement various kernels in TensorFlow.
#
# Linear Kernel:
# K(x1, x2) = t(x1) * x2
#
# Gaussian Kernel (RBF):
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Generate non-lnear data
(x_vals, y_vals) = datasets.make_circles(n_samples=350, factor=.5, noise=.1)
y_vals = np.array([1 if y == 1 else -1 for y in y_vals])
class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

# Declare batch size
batch_size = 350

# Initialize placeholders
D = 2
x_data = tf.placeholder(shape=[None, D], dtype=tf.float32) # x:[N, D]
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32) #y:[N, 1]
prediction_grid = tf.placeholder(shape=[None, D], dtype=tf.float32)

# Create variables for svm
# alpha:李航机器学习中的 支持向量系数
# alpha:[1, N]
alpha = tf.Variable(tf.random_normal(shape=[1, batch_size])) #　注意，仅有alpha是参数

# Apply kernel
# Linear Kernel
# my_kernel = tf.matmul(x_data, tf.transpose(x_data))

# Gaussian (RBF) kernel
gamma = tf.constant(50.0) # gamma 越大, 其峰越尖
x_square = tf.reduce_sum(tf.square(x_data), axis=1, keepdims=True) # dist:[N, 1]
x_x_cross = tf.matmul(x_data, tf.transpose(x_data)) # linear_kernel:[N, N]

# 类比: (xi - xj)^2 = xi^2+xj^2 - 2*xi*xj
# 计算任意两点之间的 距离 dist:[N, N]
square_dists = x_square - 2 * x_x_cross + tf.transpose(x_square) # square_dists:[N, N]
rbf_kernel = tf.exp(-gamma * tf.abs(square_dists)) #  rbf: [N, N]

# Compute SVM Model
first_term = tf.reduce_sum(alpha) # scalar
alpha_vec_cross = tf.matmul(tf.transpose(alpha), alpha) # bi*bj,  [N, N]
y_target_cross = tf.matmul(y_target, tf.transpose(y_target)) #　yi*yj, [N, N]

# 注意:此处的目标函数是对偶函数,而非原始的函数
cross_term = tf.reduce_sum(rbf_kernel * alpha_vec_cross * y_target_cross)
loss = cross_term - first_term

# w_j = sum_i{yi*alpha_i* kernel(i, j)}
# b* = yj - sum_i{yi*alpha_i* kernel(xi, xj)}
"""
下面计算b的方式有误?
w_j = y_target*tf.transpose(alpha)*tf.reduce_sum(rbf_kernel, axis=1, keepdims=True) # w:[N, 1]
bias = tf.reduce_mean(y_target - w_j)
"""
# Create Prediction Kernel
# Linear prediction kernel
# my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
x_pred_cross = tf.matmul(x_data, tf.transpose(prediction_grid))

pred_sq_dist = rA - 2*x_pred_cross + tf.transpose(rB) # [N,N]
pred_kernel = tf.exp(-gamma*tf.abs(pred_sq_dist)) # exp(-r*dist^2) , [N, N]

prediction_output = tf.matmul(tf.transpose(y_target)*alpha, pred_kernel) # [1, N]
# TODO: 这里bias的计算一直未想明白
prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output)) # ?不知为何此处这样处理
#prediction = tf.sign(prediction_output - bias) # bias好像计算有误, 暂未查到原因
#prediction = tf.sign(prediction_output)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.002)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)

    if (i + 1) % 250 == 0:
        print('Step #' + str(i + 1))
        print('Loss = ' + str(temp_loss))

# Create a mesh to plot points in
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
[grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x,
                                                     y_target: rand_y,
                                                     prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('Gaussian SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# Evaluate on new/unseen data points
# New data points:
new_points = np.array([(-0.75, -0.75),
                       (-0.5, -0.5),
                       (-0.25, -0.25),
                       (0.25, 0.25),
                       (0.5, 0.5),
                       (0.75, 0.75)])

[evaluations] = sess.run(prediction, feed_dict={x_data: x_vals,
                                                y_target: np.transpose([y_vals]),
                                                prediction_grid: new_points})

for ix, p in enumerate(new_points):
    print('{} : class={}'.format(p, evaluations[ix]))