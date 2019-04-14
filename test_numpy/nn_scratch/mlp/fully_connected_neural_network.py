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
# 为了运行此代码,需要将 nn_scratch设为 "Make directory as source root"
from cnn.mnist.read_mnist import load_mnist

# Path for the dataset file
#path = 'C:/Users/Rachneet Kaur/Desktop/UIUC/UIUC Fall 2018/IE 534 CS 598 Deep Learning/HW/Datasets/'

# MNIST dataset
#MNIST_data = h5py.File(path + 'MNISTdata.hdf5', 'r')

# Training set
#x_train = np.float32(MNIST_data['x_train'][:])  # x_train.shape = (60000, 784)
#y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))  # y_train.shape = (60000, 1)
x_train,y_train = load_mnist("../cnn/mnist/","train")
print('MNIST Training set shape =', x_train.shape)

# Testing set
#x_test = np.float32(MNIST_data['x_test'][:])  # x_test.shape = (10000, 784)
#y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))  # y_test.shape = (10000, 1)
x_test, y_test = load_mnist("../cnn/mnist/","t10k")
print('MNIST Test set shape =', x_test.shape)

#MNIST_data.close()

"""
更好的softmax计算方式是将np.sum中的每一个元素减去一个最大值,以防溢出
"""
# Defining the softmax function for the output layer
"""
def softmax_function(z):
    Z = np.exp(z) / np.sum(np.exp(z))
    return Z
"""

# z:[dim,1]
def softmax_function(z):
    #print("z shape:", z.shape)
    z_max = np.max(z)
    z_norm = z - z_max # 归一化,以防溢出
    Z = np.exp(z_norm) / np.sum(np.exp(z_norm))
    return Z

# Defining the activation function and it's derivative if flag derivative = 1
def tanh_activation(z, derivative=0):
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
input_dim = x_train.shape[1] #
# number of input features for each image = 28*28 = 784 = d
output_dim = 10  # number of output classes = k
hidden_dim = 140  # number of hidden units in the hidden layer = d_H

# Initializing the parameters for the Neural Network model
model = {}
model['W'] = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim) # [hidden_dim, input_dim] dimensional
model['b1'] = np.random.randn(hidden_dim, 1) # [hidden_dim], d_H dimensional
model['C'] = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim) # k*d_H dimensional
model['b2'] = np.random.randn(output_dim, 1) # k*1 dimensional

model_grads = copy.deepcopy(model)


EPS=1e-5
# Defining the forward step of the Neural Network model
def forward(x, y, model):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1) # y:[1, 1]
    Z = np.dot(model['W'], x) + model['b1']  # Z = Wx + b_1, d_H dimensional
    H = relu(Z)  # H = activation(Z) - d_H dimensional
    U = np.dot(model['C'], H) + model['b2']  # U = CH +b_2, k dimensional
    prob_dist = softmax_function(U) # p:[output_dim, 1]
    y_one_hot = np.zeros((output_dim,1))
    y_one_hot[y,0] =1
    loss = np.sum(-y_one_hot*np.log(prob_dist+EPS))
    # Prob_distribution of classes = F_softmax(U) - k dimensional
    return Z, H, prob_dist, loss


# Defining the backpropogation step of the Neural Network model
def backward(x, y, Z, H, prob_dist, model, model_grads):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    dZ = -1.0 * prob_dist
    dZ[y] = 1+dZ[y]
    #dZ = np.copy(prob_dist)
    #dZ[y] = 1 - dZ[y]
    # Gradient(log(F_softmax)) wrt U = Indicator Function - F_softmax
    model_grads['b2'] = dZ  # Gradient(b_2) = Gradient(log(F_softmax)) wrt U
    # k*1 dimensional
    model_grads['C'] = np.dot(dZ, np.transpose(H))  # Gradient(C) = (Gradient(log(F_softmax)) wrt U)*H^(Transpose)
    # k*d_H dimensional
    delta = np.dot(np.transpose(model['C']), dZ)  # delta = Gradient(H) = C^(Transpose)*(Gradient(log(F_softmax)) wrt U)
    # d_H dimensional
    model_grads['b1'] = np.multiply(delta, relu(Z, 1))  # Gradient(b_1) = delta.derivative of activation(Z)
    # d_H dimensional
    model_grads['W'] = np.dot(np.multiply(delta, relu(Z, 1)),
                              np.transpose(x))  # Gradient(W) = delta.derivative of activation(Z) * X^(Transpose)
    # d_H*d dimensional
    return model_grads

def update_model(model:dict, learning_rate:float, model_grads:dict):
    model['W'] = model['W'] + learning_rate * model_grads['W']  # Updating the parameters W, b_1, C and b_2 via the SGD step
    model['b1'] = model['b1'] + learning_rate * model_grads['b1']
    model['C'] = model['C'] + learning_rate * model_grads['C']
    model['b2'] = model['b2'] + learning_rate * model_grads['b2']

time1 = time.time()
learning_rate = .01
num_epochs = 15  # No. of epochs we are training the model
sample_num = len(x_train)
# Stochastic Gradient Descent algorithm
for epochs in range(num_epochs):
    # Defining the learning rate based on the no. of epochs
    if (epochs > 5):
        learning_rate = 0.001
    if (epochs > 10):
        learning_rate = 0.0001
    if (epochs > 15):
        learning_rate = 0.00001

    # Updating the parameters based on the SGD algorithm
    # 此处并非batch的训练方式
    total_correct = 0

    choose_index = np.random.permutation(range(sample_num))
    losses=[]
    for n in range(sample_num):
        #n_random = randint(0, sample_num - 1)
        index = choose_index[n]
        x = x_train[index][:] # x:[input_dim,]
        y = y_train[index] # y:[,]
        Z, H, prob_dist, loss = forward(x, y, model)
        losses.append(loss)
        model_grads = backward(x, y, Z, H, prob_dist, model, model_grads)
        update_model(model, learning_rate, model_grads)
        if n%5000 ==0:
            print("iter:", n, "avg loss:", np.mean(losses))
            losses=[]

        # record predict labels
        prediction = np.argmax(prob_dist)
        if (prediction == y):
            total_correct += 1
    print('In epoch ', epochs, ', accuracy in training set = ', total_correct / np.float(len(x_train)))

time2 = time.time()
print('Estimated time to train the model = ', time2 - time1, ' seconds')

test_accuracy = compute_accuracy(x_test, y_test, model)
print('Accuracy in testing set =', test_accuracy)

"""
type: train label magic: 2049 label number: 60000
type: train image magic: 2051 image number: 60000
MNIST Training set shape = (60000, 784)
type: t10k label magic: 2049 label number: 10000
type: t10k image magic: 2051 image number: 10000
MNIST Test set shape = (10000, 784)
In epoch  0 , accuracy in training set =  0.9304166666666667
In epoch  1 , accuracy in training set =  0.9700666666666666
In epoch  2 , accuracy in training set =  0.9770833333333333
In epoch  3 , accuracy in training set =  0.9816166666666667
In epoch  4 , accuracy in training set =  0.9843833333333334
In epoch  5 , accuracy in training set =  0.9869166666666667
"""