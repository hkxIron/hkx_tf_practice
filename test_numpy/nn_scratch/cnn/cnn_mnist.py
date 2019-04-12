"""
Problem 2: Implementation of Convolution Neural Network from scratch using numpy

Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).
The convolution network should have a single hidden layer with multiple channels. Target accuracy on the test set: 94%

Implementation of Convolution Neural Networks on MNIST dataset
Rachneet Kaur, rk4
Accuracy in testing set = 0.9735

"""
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:01:02 2018
@author: Rachneet Kaur
"""

# Result of testing accuracy = 97.35% with 10 epochs of SGD

# Library imports
import numpy as np
import h5py
import time
import copy
from random import randint
import itertools

# Path for the dataset file
path = 'C:/Users/Rachneet Kaur/Desktop/UIUC/UIUC Fall 2018/IE 534 CS 598 Deep Learning/HW/Datasets/'

# MNIST dataset
MNIST_data = h5py.File(path + 'MNISTdata.hdf5', 'r')
d = 28  # number of input features for each image = 28*28  = d * d

# Training set
x_train = np.float32(MNIST_data['x_train'][:])  # x_train.shape = (60000, 784)
x_train = np.array([x.reshape(d, d) for x in x_train])  # Reshaping the image in a matrix format
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))  # y_train.shape = (60000, 1)
y_train = [y.reshape(-1, 1) for y in y_train]
print('MNIST Training set shape =', x_train.shape)

# Testing set
x_test = np.float32(MNIST_data['x_test'][:])  # x_test.shape = (10000, 784)
x_test = np.array([x.reshape(d, d) for x in x_test])  # Reshaping the image in a matrix format
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))  # y_test.shape = (10000, 1)
y_test = [y.reshape(-1, 1) for y in y_test]
print('MNIST Test set shape =', x_test.shape)
MNIST_data.close()


# Defining the softmax function for the output layer
def softmax_function(z):
    softmax = np.exp(z) / np.sum(np.exp(z))
    return softmax


# Defining the convolution function
def convolution(X, K, iteratable, d_y, d_x):  # X - image, K - filter, (d_x, d_y, channel) = Dimensions of filter K
    conv_Z = np.array([np.tensordot(K[:, :, tuple_ijk[2]],
                                    X[tuple_ijk[0]:tuple_ijk[0] + d_y, tuple_ijk[1]:tuple_ijk[1] + d_x],
                                    axes=((0, 1), (0, 1))) for tuple_ijk in iteratable])
    return conv_Z


# Defining the activation function and it's derivative if flag derivative = 1
def activation(Z, derivative=0):
    if (derivative == 1):
        return 1.0 - np.tanh(Z) ** 2  # Derivative of tanh(z) applied elementwise
    else:
        return np.tanh(Z)  # tanh(z) as activation function applied elementwise


# Function to compute the accuracy on the testing dataset
def compute_accuracy(x_series, y_series, model):
    total_correct = 0
    for index in range(len(x_series)):
        y = y_series[index]  # True label
        x = x_series[index][:]  # Input
        Z, H, p = forward(x, y, model)
        prediction = np.argmax(p)  # Predicting the label based on the input
        if (prediction == y):  # Checking if True label == Predicted label
            total_correct += 1
    accuracy = total_correct / np.float(len(x_series))
    return accuracy


# Shape parameters for the layers
num_outputs = 10  # number of output classes = k
# Dimensions for the Kernel = d_y * d_x * C
d_y = 5
d_x = 5
channel = 5  # No. of channels C
# dimensions of hidden units in the hidden layer
num_hidden_x = d - d_y + 1
num_hidden_y = d - d_x + 1

# Initializing the parameters for the Convolution Neural Network model
model = {}
model['K'] = np.random.randn(d_y, d_x, channel) / np.sqrt(d_x * d_y)
# K = d_y *d_x * C dimensional
model['W'] = np.random.randn(num_outputs, num_hidden_x, num_hidden_y, channel) / np.sqrt(num_hidden_x * num_hidden_y)
# W = k*(d-d_y+1)*(d-d_x+1)*C dimensional
model['b'] = np.random.randn(num_outputs, 1)
# b = k*1 dimensional

model_grads = copy.deepcopy(model)

# Defining the iteratables for the convolution function
l1 = range(num_hidden_x)
l2 = range(num_hidden_y)
l3 = range(channel)
iteratable_forward = list(itertools.product(l1, l2, l3))

i1 = range(d_y)
i2 = range(d_x)
iteratable_backward = list(itertools.product(i1, i2, l3))


# Defining the forward step of the Convolution Neural Network model
def forward(x, y, model):
    Z = convolution(x, model['K'], iteratable_forward, d_y, d_x).reshape(num_hidden_x, num_hidden_y, channel)
    # Z = X convolution K = d-d_y+1*d-d_x+1*channel dim.
    H = activation(Z)  # H = activation(Z) - (d-d_y+1)*(d-d_x+1)*channel dimensional
    U = np.tensordot(model['W'], H, axes=((1, 2, 3), (0, 1, 2))).reshape(-1, 1) + model[
        'b']  # U = W.H + b - k dimensional
    prob_dist = softmax_function(U)  # Prob_distribution of classes = F_softmax(U) - k dimensional
    return Z, H, prob_dist


# Defining the backpropogation step of the Convolution Neural Network model
def backward(x, y, Z, H, prob_dist, model, model_grads):
    dZ = -1.0 * prob_dist
    dZ[y] = (dZ[y] + 1.0)
    dZ = -dZ
    # Gradient(log(F_softmax)) wrt U = Indicator Function - F_softmax
    model_grads['b'] = dZ  # Gradient(b) = Gradient(log(F_softmax)) wrt U
    # Gradient_b = k*1 dimensional
    model_grads['W'] = np.tensordot(dZ.T, H, axes=0)[0]
    # Gradient_W = k*(d-d_y+1)*(d-d_x+1)*C dimensional
    delta = np.tensordot(dZ.T, model['W'], axes=1)[
        0]  # delta_{i,j,p} = Gradient(H) = (Gradient(log(F_softmax)) wrt U)*W_{:,i,j,p}
    # delta = (d-d_y+1)* (d-d_x+1)* C dimensional
    model_grads['K'] = convolution(x, np.multiply(delta, activation(Z, 1)), iteratable_backward, d - d_y + 1,
                                   d - d_x + 1).reshape(d_y, d_x, channel)
    # Gradient(W) = X convolution delta.derivative of activation(Z)
    # Using the dimensions of np.multiply(delta, activation(Z, 1)) as an input to the convolution function
    # model_grads['K'] = d_y*d_x*C dimensional
    return model_grads


LR = .01
num_epochs = 10  # No. of epochs we are training the model

# Stochastic Gradient Descent algorithm
for epochs in range(num_epochs):
    time1 = time.time()
    # Defining the learning rate based on the no. of epochs
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001

    # Updating the parameters based on the SGD algorithm
    total_correct = 0
    for n in range(len(x_train)):
        n_random = randint(0, len(x_train) - 1)  # SGD step
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z, H, prob_dist = forward(x, y, model)
        prediction = np.argmax(prob_dist)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, Z, H, prob_dist, model, model_grads)
        model['W'] = model['W'] - LR * model_grads['W']  # Updating the parameters W, b, and K via the SGD step
        model['b'] = model['b'] - LR * model_grads['b']
        model['K'] = model['K'] - LR * model_grads['K']
    print('In epoch ', epochs, ', accuracy in training set = ', total_correct / np.float(len(x_train)))

# Calculating the test accuracy
test_accuracy = compute_accuracy(x_test, y_test, model)
print('Accuracy in testing set =', test_accuracy)