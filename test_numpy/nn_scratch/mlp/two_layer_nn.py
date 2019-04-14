#coding:utf-8
import numpy as np

def sigmoid(x, deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

#4*3 
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
batch, x_dim = X.shape
print("batch:", batch, " x_dim:",x_dim)

# [batch,1],4*1
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
h0_dim = 4
w0 = 2 * np.random.random((x_dim, h0_dim)) - 1  # x_dim*h0_dim,即w
w1 = 2 * np.random.random((h0_dim, 1)) - 1  # h0_dim*1

N = 60000
for j in range(N):
    """
    Feed forward through layers 0, 1, and 2
    X:[batch, x_dim]
    w0:[x_dim, h0_dim]
    w1:[h0_dim, 1]
    y_hat:[batch, 1]
    
    z0 = x
    a0 = z0*w0 + b
    z1 = sigmoid(a0)
    a1 = z1*w1 + bias
    y_hat = sigmoid(a1) 
    
    loss:
    L = 1/2*(y-y_hat)^2

    反向传播:
    第二层
    dL/dy_hat = y_hat - y 
    dy_hat/da1 = y_hat*(1-y_hat) 
    
    dL/da1 = dL/dy_hat*dy_hat/da1 = (y_hat-y)*y_hat*(1-y_hat)
    da1/dw1 = z1
    da1/dz1 = w1
    =>
    dL/dw1=dL/da1*da1/dw1 = (y_hat - y)*y_hat*(1-y_hat)*z1
    dL/dz1= dL/da1*da1/dz1 = dL/da1* w1
    
    第一层 
    dz1/da0 = z1*(1-z1)
    da0/dw0 = z0
    da0/dz0 = w0
    dL/dw0 = dL/dz1*dz1/da0*da0/dw0 = dL/dz1*z1*(1-z1)*z0
    dL/dz0 = dL/dz1*dz1/da0*da0/dz0
    """
    z0 = X #4*3
    # z0:[batch, x_dim]
    # w0:[x_dim, h0_dim]
    # z1:[batch, h0_dim]
    a0 = np.dot(z0, w0)
    z1 = sigmoid(a0)

    # z1:[batch, h0_dim]
    # w1:[h0_dim, 1]
    # a1:[batch, 1]
    # y_hat:[batch, 1]
    a1 = np.dot(z1, w1)
    y_hat = sigmoid(a1)

    loss = np.mean((y-y_hat)**2)
    if (j% 10000) == 0:
        #print("Error:" + str(np.mean(np.abs(l2_error))))
        print("square loss:", loss)

    dL__dy_hat = y_hat - y # __表示除
    dy_hat__da1 = sigmoid(y_hat, deriv=True) # [batch, 1]
    dL__da1 = dL__dy_hat * dy_hat__da1 # [batch, 1]

    # w1:[h0_dim,1]
    # dL__da1:[batch, 1]
    # dL__dz1:[batch, h0_dim]
    dL__dz1 = dL__da1.dot(w1.T)

    # z1:[batch, h0_dim]
    # dz1__da0:[batch, h0_dim]
    # dL__da0:[batch, h0_dim]
    dz1__da0 = sigmoid(z1, deriv=True)
    dL__da0 = dL__dz1 * dz1__da0

    # z1:[batch, h0_dim]
    # dL__da1:[batch, 1]
    # dL__dw1:[h0_dim, 1]
    dL__dw1 = z1.T.dot(dL__da1)
    # z0:[batch, x_dim]
    # dL__da0:[batch, h0_dim]
    # g0:[x_dim, h0_dim]
    dL__dw0 = z0.T.dot(dL__da0)

    w1 -= dL__dw1
    w0 -= dL__dw0


print("syn0:")
print(w0)
print("syn1:")
print(w1)

"""
batch: 4  x_dim: 3
square loss: 0.2475932677010128
square loss: 7.537818754812717e-05
square loss: 3.4261210615712536e-05
square loss: 2.1895641392008393e-05
square loss: 1.600802541539428e-05
square loss: 1.2582191960640921e-05
syn0:
[[ 4.6013571   4.17197193 -6.30956245 -4.19745118]
 [-2.58413484 -5.81447929 -6.60793435 -3.68396123]
 [ 0.97538679 -2.02685775  2.52949751  5.84371739]]
syn1:
[[ -6.96765763]
 [  7.14101949]
 [-10.31917382]
 [  7.86128405]]
"""