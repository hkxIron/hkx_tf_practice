"""

Problem 1: Implementation of fully connected neural network from scratch using numpy
Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. Target accuracy on the test set: 97-98%


https://github.com/hkxIron/CS598-Deep-Learning-MPs/blob/master/MP1_FeedForwardWithoutPytorch/NN_MNIST_IE534HW1.py

"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 10:01:02 2018
@author: Rachneet Kaur
"""

# Library imports
import numpy as np
import h5py
import time
import copy
from random import randint

# Path for the dataset file
path = 'C:/Users/Rachneet Kaur/Desktop/UIUC/UIUC Fall 2018/IE 534 CS 598 Deep Learning/HW/Datasets/'

# MNIST dataset
MNIST_data = h5py.File(path + 'MNISTdata.hdf5', 'r')

# Training set
x_train = np.float32(MNIST_data['x_train'][:])  # x_train.shape = (60000, 784)
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))  # y_train.shape = (60000, 1)
print('MNIST Training set shape =', x_train.shape)

# Testing set
x_test = np.float32(MNIST_data['x_test'][:])  # x_test.shape = (10000, 784)
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))  # y_test.shape = (10000, 1)
print('MNIST Test set shape =', x_test.shape)

MNIST_data.close()


# Defining the softmax function for the output layer
def softmax_function(z):
    Z = np.exp(z) / np.sum(np.exp(z))
    return Z


# Defining the activation function and it's derivative if flag derivative = 1
def activation(z, derivative=0):
    if (derivative == 1):
        return 1.0 - np.tanh(z) ** 2  # Derivative of tanh(z)
    else:
        return np.tanh(z)  # tanh(z) as activation function


def relu(Z2, derivative=0):
    if (derivative == 1):
        Z2[Z2 > 0] = 1
        Z2[Z2 <= 0] = 0
    else:
        Z2[Z2 < 0] = 0
    return Z2


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
num_inputs = x_train.shape[1]
# number of input features for each image = 28*28 = 784 = d
num_outputs = 10  # number of output classes = k
num_hidden = 140  # number of hidden units in the hidden layer = d_H

# Initializing the parameters for the Neural Network model
model = {}
model['W'] = np.random.randn(num_hidden, num_inputs) / np.sqrt(num_inputs)
# d_H*d dimensional
model['b1'] = np.random.randn(num_hidden, 1)
# d_H dimensional
model['C'] = np.random.randn(num_outputs, num_hidden) / np.sqrt(num_hidden)
# k*d_H dimensional
model['b2'] = np.random.randn(num_outputs, 1)
# k*1 dimensional

model_grads = copy.deepcopy(model)


# Defining the forward step of the Neural Network model
def forward(x, y, model):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    Z = np.dot(model['W'], x) + model['b1']  # Z = Wx + b_1 - d_H dimensional
    H = relu(Z)  # H = activation(Z) - d_H dimensional
    U = np.dot(model['C'], H) + model['b2']  # U = CH +b_2 - k dimensional
    prob_dist = softmax_function(U)
    # Prob_distribution of classes = F_softmax(U) - k dimensional
    return Z, H, prob_dist


# Defining the backpropogation step of the Neural Network model
def backward(x, y, Z, H, prob_dist, model, model_grads):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    dZ = -1.0 * prob_dist
    dZ[y] = dZ[y] + 1.0
    # Gradient(log(F_softmax)) wrt U = Indicator Function - F_softmax
    model_grads['b2'] = dZ  # Gradient(b_2) = Gradient(log(F_softmax)) wrt U
    # k*1 dimensional
    model_grads['C'] = np.dot(dZ, np.transpose(H))  # Gradient(C) = (Gradient(log(F_softmax)) wrt U)*H^(Transpose)
    # k*d_H dimensional
    delta = np.dot(np.transpose(model['C']), dZ)  # delta = Gradient(H) = C^(Transpose)*(Gradient(log(F_softmax)) wrt U)
    # d_H dimensional
    model_grads['b1'] = np.multiply(delta, activation(Z, 1))  # Gradient(b_1) = delta.derivative of activation(Z)
    # d_H dimensional
    model_grads['W'] = np.dot(np.multiply(delta, relu(Z, 1)),
                              np.transpose(x))  # Gradient(W) = delta.derivative of activation(Z) * X^(Transpose)
    # d_H*d dimensional
    return model_grads


time1 = time.time()
LR = .01
num_epochs = 15  # No. of epochs we are training the model
# Stochastic Gradient Descent algorithm
for epochs in range(num_epochs):
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
        n_random = randint(0, len(x_train) - 1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z, H, prob_dist = forward(x, y, model)
        prediction = np.argmax(prob_dist)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, Z, H, prob_dist, model, model_grads)
        model['W'] = model['W'] + LR * model_grads['W']  # Updating the parameters W, b_1, C and b_2 via the SGD step
        model['b1'] = model['b1'] + LR * model_grads['b1']
        model['C'] = model['C'] + LR * model_grads['C']
        model['b2'] = model['b2'] + LR * model_grads['b2']
    print('In epoch ', epochs, ', accuracy in training set = ', total_correct / np.float(len(x_train)))
time2 = time.time()
print('Estimated time to train the model = ', time2 - time1, ' seconds')

test_accuracy = compute_accuracy(x_test, y_test, model)
print('Accuracy in testing set =', test_accuracy)