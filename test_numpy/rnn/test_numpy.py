#coding:utf-8
import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
np.random.seed(0);
W=np.random.randn(3,4)
x=np.random.randn(4,1)
z = 1/(1 + np.exp(-np.dot(W, x))) # forward pass,3*1
dx = np.dot(W.T, z*(1-z)) # backward pass: local gradient for x
dW = np.outer(z*(1-z), x) # backward pass: local gradient for W
print dW.shape

print "-"*10
x=np.array([1,2,-2,0])
np.reshape(x, (4,1))
W[0]=np.array([0,-1,1,0])
print "W",W
z = np.maximum(0, np.dot(W, x)) # forward pass
print "z",z
dW = np.outer(z > 0, x) # backward pass: local gradient for W
print dW

