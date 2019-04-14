"""

Problem 1: Implementation of fully connected neural network from scratch using numpy
Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. Target accuracy on the test set: 97-98%


https://github.com/hkxIron/CS598-Deep-Learning-MPs/blob/master/MP1_FeedForwardWithoutPytorch/NN_MNIST_IE534HW1.py
本代码基本由hukexin按照自己的思路进行了重构

"""

# -*- coding: utf-8 -*-

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

# z:[batch, label_num]
def softmax_function(z):
    #print("z shape:", z.shape)
    z_max = np.max(z, axis=1, keepdims=True)
    z_norm = z - z_max # 归一化,以防溢出
    Z = np.exp(z_norm) / np.sum(np.exp(z_norm), axis=1, keepdims=True)
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




# Shape parameters for the layers
input_dim = x_train.shape[1] #
# number of input features for each image = 28*28 = 784 = d
hidden_dim = 140  # number of hidden units in the hidden layer = d_H
output_dim = 10  # number of output classes = k

# Initializing the parameters for the Neural Network model
model = {}
model['W'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim) # [input_dim, hidden_dim]
model['b1'] = np.random.randn(1, hidden_dim) # [1, hidden_dim], d_H dimensional

model['C'] = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim) # [hidden_dim, output_dim]
model['b2'] = np.random.randn(1, output_dim) # [1, output_dim]

model_grads = copy.deepcopy(model)

EPS=1e-5
# Defining the forward step of the Neural Network model
def forward(x, y, model):
    # x:[batch_size,input_dim]
    # y:[batch_size,label_dim
    # w:[input_dim, hidden_dim], b1:[1, hidden_dim]
    # Z: [batch_size, hidden_dim]
    Z = np.dot(x, model['W']) + model['b1']  # Z = Wx + b_1, d_H dimensional
    # Z: [batch_size, hidden_dim]
    # H: [batch_size, hidden_dim]
    H = relu(Z)
    # C: [hidden_dim, output_dim]
    # b2: [1, output_dim]
    # U: [batch_size, output_dim]
    U = np.dot(H, model['C']) + model['b2']  # U = CH +b_2, k dimensional
    # prob_dist: [batch_size, output_dim]
    prob_dist = softmax_function(U)
    # cross entropy loss, sum{-y_i*log(p_i)}
    loss = np.mean(np.sum(-y*np.log(prob_dist+EPS),axis=1))
    return Z, H, prob_dist, loss


# Defining the backpropogation step of the Neural Network model
def backward(x, y, Z, H, prob_dist, model, model_grads):
    """
    softmax:
    y(xk) = exp(xk)/ sum_k' (  exp(xk')  ),计算时可以除以一个最大的数,以防溢出

    注意，只有在k=k'时，值为 yk*(1-yk),否则为 -yk*yk'，切记切记！以前一直认识有误区!
    k==k':  yk * (1-yk)
    k!=k':  -yk * yk'

    forward:
    # x:[batch, input_dim], w:[input_dim, hidden_dim],
    # b1:[1, hidden_dim], Z:[batch, hidden_dim]
    Z = X*w + b1
    H = relu(Z)
    U = H * C + b2
    p = softmax(U)
    p = [p_1, p_2, ... , p_k]
    loss:
    L = - sum_k(y_k* log(p_k))

    z_k = softmax(a_k)
    推导可知:
    dL/da_k = z_k - y_k
    即
    dL/dU = z - y
    dL/db2 = dL/dU *dU/db2 = dL/dU
    """

    #dZ =  prob_dist
    #dZ[y] = 1.0 - dZ[y]

    # prob_dist:[batch, output_dim]
    # y:[batch, output_dim]
    # dL_du: [batch, output_dim]
    dL__dU = prob_dist - y
    model_grads['b2'] = dL__dU
    # H:[batch, hidden_dim]
    # C:[hidden_dim, output_dim]
    # dL/dC = dL/dU*dU/dC = dL/dU*H
    # dL/dC = H^T*dL/du
    dL__dC = np.dot(np.transpose(H), dL__dU)/ batch_size
    model_grads['C'] = dL__dC

    # dL/dH = dL/dU*dU/dH = dL/dU*C
    # dL__dH = C^T * dL__U
    # C: [hidden_dim, output_dim]
    # dL_dU: [batch, output_dim]
    # dL_dH: [batch, hidden_dim]
    dL__dH = np.dot(dL__dU, np.transpose(model["C"]))/batch_size

    # dL/dZ = dL/dH*dH/dz
    # dL_dZ: [batch, hidden_dim]
    dL__dZ = dL__dH*relu(Z, derivative=1) # 注意:relu里的是Z,而不是dL__dH
    # dL_dH: [batch, hidden_dim]
    dL__db1 = dL__dZ
    model_grads['b1'] = dL__db1
    # dL_dH: [batch, hidden_dim]
    # w:[input_dim, hidden_dim]
    # X:[batch, input_dim]
    # dL__dw:[input_dim, hidden_dim]
    # dL__dw = dL/dZ*dZ/dx = x^T*dL/dZ
    dL__dw = np.dot(np.transpose(x), dL__dZ)/batch_size # w的更新与batch_size无关
    model_grads['W'] = dL__dw
    # d_H*d dimensional
    return model_grads

def update_model(model:dict, learning_rate:float, model_grads:dict):
    model['W'] -=  learning_rate * model_grads['W'] # Updating the parameters W, b_1, C and b_2 via the SGD step
    # b1:[batch, hidden_dim]
    # b1:[1, hidden_dim]
    model['b1'] -= learning_rate * np.mean(model_grads['b1'],axis=0)
    model['C'] -=  learning_rate * model_grads['C']
    # b2:[1, input_dim]
    model['b2'] -= learning_rate * np.mean(model_grads['b2'],axis=0)

time1 = time.time()
learning_rate = .01
num_epochs = 15  # No. of epochs we are training the model
sample_num = len(x_train)
batch_size = 20

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
    batch_no = np.int(np.round(sample_num//batch_size + 0.5))
    choose_index = np.random.permutation(range(batch_no))

    for n in range(batch_no):
        #n_random = randint(0, sample_num - 1)
        index = choose_index[n]*batch_size
        x = x_train[index:index+batch_size,:] # x:[batch_size,input_dim]
        y = y_train[index:index+batch_size] # y:[batch_size,1]
        # Z: [batch_size, hidden_dim]
        # H: [batch_size, hidden_dim]
        y_one_hot = np.zeros((batch_size, output_dim))
        for ind in range(batch_size):
            y_one_hot[ind][y[ind]] = 1

        Z, H, prob_dist, loss = forward(x, y_one_hot, model)
        # record predict labels
        prediction = np.argmax(prob_dist, axis=1)
        total_correct += np.sum(prediction == y)
        if n%5000 ==0: print("epoch:", epochs,"iter:", n, "loss:", loss)
        model_grads = backward(x, y_one_hot, Z, H, prob_dist, model, model_grads)
        update_model(model, learning_rate, model_grads)
    print('In epoch ', epochs, ', accuracy in training set = ', total_correct / np.float(len(x_train)))

time2 = time.time()
print('Estimated time to train the model = ', time2 - time1, ' seconds')

# Function to compute the accuracy on the testing dataset
def compute_accuracy(x_series, y_series, model):
    total_correct = 0
    for index in range(len(x_series)):
        y = y_series[index]  # True label
        y_one_hot = np.zeros((1, output_dim))
        y_one_hot[0, y] = 1
        x = x_series[index][:]  # Input
        Z, H, p, loss = forward(x, y_one_hot, model)
        prediction = np.argmax(p)  # Predicting the label based on the input
        if (prediction == y):  # Checking if True label == Predicted label
            total_correct += 1
    accuracy = total_correct / np.float(len(x_series))
    return accuracy

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