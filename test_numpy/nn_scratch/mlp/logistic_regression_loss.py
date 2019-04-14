# coding:utf-8
# author:kexinhu

import numpy as np

# sigmoid function
def sigmoid(x, deriv=False):
    if(deriv==True):
        return x*(1-x) # d sigmod = sigmoid*(1-sigmoid)
    return 1/(1+np.exp(-x))
    
# input dataset
# X:[batch, x_dim]
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset
# y:[batch, 1]
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
# syn0:[x_dim, 1]

"""

简单的 logistic regression:
y_hat = sigmoid(X*w + b)

对比使用不同的loss所带来的预测误差变化,实验表明,cross-entropy-loss比square-loss使算法收敛更快

square loss
square_loss: 0.3099458499092816
square_loss: 0.008450761991644287
square_loss: 0.0039182395524627375
square_loss: 0.0025147089298240507
square_loss: 0.0018417284716549702
square_loss: 0.0014490820314633829
square_loss: 0.0011926100235693606
square_loss: 0.0010122816435312347
square_loss: 0.0008787359406642932
square_loss: 0.0007759466714038695
Output After Training:
[[0.03178421]
 [0.02576499]
 [0.97906682]
 [0.97414645]]
cross entropy loss
cross_entropy_loss: 1.0223746017260429
cross_entropy_loss: 0.012723497064175322
cross_entropy_loss: 0.006316959295539169
cross_entropy_loss: 0.0041956476561908875
cross_entropy_loss: 0.003138824655193752
cross_entropy_loss: 0.0025061874912061617
cross_entropy_loss: 0.0020851068770820947
cross_entropy_loss: 0.001784694363905895
cross_entropy_loss: 0.001559594039694303
cross_entropy_loss: 0.001384646275310756
Output After Training:
[[0.00181019]
 [0.00120309]
 [0.99919858]
 [0.99879392]]
"""
print("square loss")
#x_dim=3
batch_size, x_dim = X.shape
N = 1000
# w:[x_dim, 1]
w = 2 * np.random.random((x_dim, 1)) - 1
for iter in range(N):
    """
    forward propagation
    X:[batch, x_dim]
    w:[x_dim, 1]
    y_hat:[batch, 1]
    a = x*w + b
    y_hat = sigmoid(a)
    
    loss = 1/2*(y-y_hat)^2
    """
    y_hat = sigmoid(np.dot(X, w))

    # loss: L = 1/2*(y-y_hat)^2
    cross_entropy_loss = np.mean((y - y_hat) ** 2)
    if iter%100 ==0:
        print("square_loss:", cross_entropy_loss)
    # dL/dy_hat = (y-y_hat) * (-1) = y_hat - y
    # y_hat:[batch, 1]
    l1_error = y_hat - y

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1

    # dy_hat/da = y_hat*(1-y_hat)
    # dL/da = dL/dy_hat* dy_hat/da = (y_hat-y)* y_hat*(1-y_hat), 注意:这里的梯度会减小,比cross-entropy-loss里的小很多
    # l1_error:[batch, 1]
    # y_hat:[batch, 1]
    # l1_delta:[batch, 1]
    l1_delta = l1_error * sigmoid(y_hat, True)

    # dL/dw = dL/da * da/dw = dl/da*x
    # X:[batch, x_dim]
    # l1_delta:[batch, 1]
    # g: [x_dim, 1]
    g = np.dot(X.T, l1_delta)/batch_size

    # update weights
    # w: [x_dim, 1]
    w -= g

print("Output After Training:")
print(y_hat)

print("cross entropy loss")
eps = 1e-5
w = 2 * np.random.random((3, 1)) - 1
for iter in range(N):
    # forward propagation
    # X:[batch, x_dim]
    # w:[x_dim, 1]
    # y_hat:[batch, 1]
    # a = x*w + b
    # y_hat = sigmoid(a)
    y_hat = sigmoid(np.dot(X, w))

    # loss: L = - [ylog(p(x)) + (1-y)log( 1-p(x) )]
    # dL/dy_hat = (y -y_hat) /[ y_hat(1-y_hat) ]
    # dL/da = dL/dy_hat* dy_hat/da =
    # (y -y_hat) /[ y_hat(1-y_hat) ] *  dy_hat/da = y_hat - y

    cross_entropy_loss = -np.mean((y*np.log(y_hat) + (1-y)*np.log(1-y_hat)))
    if iter%100 ==0:
        print("cross_entropy_loss:", cross_entropy_loss)
    # how much did we miss?
    l1_delta = y_hat - y

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1

    # X:[batch, x_dim]
    # l1_delta:[batch, 1]
    # g:[x_dim,1]
    # dL/dw = dL/da*da/dw = (y-y_hat)*x
    g = np.dot(X.T, l1_delta)/batch_size

    # update weights
    w -= g

print("Output After Training:")
print(y_hat)
