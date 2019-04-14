"""
Problem 2: Implementation of Convolution Neural Network from scratch using numpy

Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch).
The convolution network should have a single hidden layer with multiple channels. Target accuracy on the test set: 94%

Implementation of Convolution Neural Networks on MNIST dataset
Rachneet Kaur, rk4
Accuracy in testing set = 0.9735

该代码速度极慢

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
# 为了运行此代码,需要将 nn_scratch设为 "Make directory as source root"
from mnist.read_mnist import load_mnist

# Path for the dataset file
#path = 'C:/Users/Rachneet Kaur/Desktop/UIUC/UIUC Fall 2018/IE 534 CS 598 Deep Learning/HW/Datasets/'

# MNIST dataset
# MNIST_data = h5py.File(path + 'MNISTdata.hdf5', 'r')
d = 28  # number of input features for each image = 28*28  = d * d
EPS=1e-5

# Training set
#x_train = np.float32(MNIST_data['x_train'][:])  # x_train.shape = (60000, 784)
#x_train = np.array([x.reshape(d, d) for x in x_train])  # Reshaping the image in a matrix format
#y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))  # y_train.shape = (60000, 1)
x_train,y_train = load_mnist("mnist/","train")
# x_train:[batch, width, height]
x_train = np.array([x.reshape(d, d) for x in x_train])  # Reshaping the image in a matrix format
# y_train:list of [1,1]
y_train = [y.reshape(-1, 1) for y in y_train]

x_test, y_test = load_mnist("mnist/","t10k")
x_test = np.array([x.reshape(d, d) for x in x_test])  # Reshaping the image in a matrix format
y_test = [y.reshape(-1, 1) for y in y_test]

print('MNIST Training set shape =', x_train.shape)
print('MNIST Test set shape =', x_test.shape)

# Testing set
#x_test = np.float32(MNIST_data['x_test'][:])  # x_test.shape = (10000, 784)
#x_test = np.array([x.reshape(d, d) for x in x_test])  # Reshaping the image in a matrix format
#y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))  # y_test.shape = (10000, 1)
#MNIST_data.close()


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
def tanh_activation(Z, derivative=0):
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
        Z, H, p, loss = forward(x, y, model)
        prediction = np.argmax(p)  # Predicting the label based on the input
        if (prediction == y):  # Checking if True label == Predicted label
            total_correct += 1
    accuracy = total_correct / np.float(len(x_series))
    return accuracy


# Shape parameters for the layers
output_dim = 10  # number of output classes = k
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
model['W'] = np.random.randn(output_dim, num_hidden_x, num_hidden_y, channel) / np.sqrt(num_hidden_x * num_hidden_y)
# W = k*(d-d_y+1)*(d-d_x+1)*C dimensional
model['b'] = np.random.randn(output_dim, 1)
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
    H = tanh_activation(Z)  # H = activation(Z) - (d-d_y+1)*(d-d_x+1)*channel dimensional
    U = np.tensordot(model['W'], H, axes=((1, 2, 3), (0, 1, 2))).reshape(-1, 1) + model['b']  # U = W.H + b - k dimensional
    prob_dist = softmax_function(U)  # Prob_distribution of classes = F_softmax(U) - k dimensional
    y_one_hot = np.zeros((output_dim,1))
    y_one_hot[y,0] =1
    loss = np.sum(-y_one_hot*np.log(prob_dist+EPS))
    return Z, H, prob_dist, loss


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
    delta = np.tensordot(dZ.T, model['W'], axes=1)[0]  # delta_{i,j,p} = Gradient(H) = (Gradient(log(F_softmax)) wrt U)*W_{:,i,j,p}
    # delta = (d-d_y+1)* (d-d_x+1)* C dimensional
    model_grads['K'] = convolution(x, np.multiply(delta, tanh_activation(Z, 1)),
                                   iteratable_backward,
                                   d - d_y + 1,
                                   d - d_x + 1).reshape(d_y, d_x, channel)
    # Gradient(W) = X convolution delta.derivative of activation(Z)
    # Using the dimensions of np.multiply(delta, activation(Z, 1)) as an input to the convolution function
    # model_grads['K'] = d_y*d_x*C dimensional
    return model_grads


learning_rate = .01
num_epochs = 10  # No. of epochs we are training the model

# Stochastic Gradient Descent algorithm
for epochs in range(num_epochs):
    time1 = time.time()
    # Defining the learning rate based on the no. of epochs
    if (epochs > 5):
        learning_rate = 0.001
    if (epochs > 10):
        learning_rate = 0.0001
    if (epochs > 15):
        learning_rate = 0.00001

    # Updating the parameters based on the SGD algorithm
    total_correct = 0
    losses=[]
    acc = []
    for n in range(len(x_train)):
        n_random = randint(0, len(x_train) - 1)  # SGD step
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z, H, prob_dist, loss = forward(x, y, model)
        prediction = np.argmax(prob_dist)
        losses.append(loss)
        acc.append(prediction == y)
        if n%100 ==0:
            print("iter:", n, "avg loss:", np.mean(losses), " avg acc:", np.mean(acc))
            losses=[]
            acc = []
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, Z, H, prob_dist, model, model_grads)
        model['W'] = model['W'] - learning_rate * model_grads['W']  # Updating the parameters W, b, and K via the SGD step
        model['b'] = model['b'] - learning_rate * model_grads['b']
        model['K'] = model['K'] - learning_rate * model_grads['K']
    print('In epoch ', epochs, ', accuracy in training set = ', total_correct / np.float(len(x_train)))

# Calculating the test accuracy
test_accuracy = compute_accuracy(x_test, y_test, model)
print('Accuracy in testing set =', test_accuracy)

"""
type: train label magic: 2049 label number: 60000
type: train image magic: 2051 image number: 60000
type: t10k label magic: 2049 label number: 10000
type: t10k image magic: 2051 image number: 10000
MNIST Training set shape = (60000, 28, 28)
MNIST Test set shape = (10000, 28, 28)
iter: 0 avg loss: 3.862474000413405  avg acc: 0.0
iter: 100 avg loss: 1.8182736719734638  avg acc: 0.4
iter: 200 avg loss: 1.0899630674948912  avg acc: 0.64
iter: 300 avg loss: 0.977622220964676  avg acc: 0.71
iter: 400 avg loss: 0.6573021473395757  avg acc: 0.81
iter: 500 avg loss: 0.6043973564652194  avg acc: 0.81
iter: 600 avg loss: 0.5268250558857629  avg acc: 0.83
iter: 700 avg loss: 0.505714093710708  avg acc: 0.82
iter: 800 avg loss: 0.6143088166739408  avg acc: 0.78
iter: 900 avg loss: 0.5209232961652669  avg acc: 0.77
iter: 1000 avg loss: 0.5167780041912328  avg acc: 0.87
iter: 1100 avg loss: 0.7979195793203246  avg acc: 0.75
iter: 1200 avg loss: 0.8843380728479993  avg acc: 0.74
iter: 1300 avg loss: 0.8316806449867536  avg acc: 0.8
iter: 1400 avg loss: 0.6758073308795899  avg acc: 0.71
iter: 1500 avg loss: 0.6344304292750593  avg acc: 0.85
iter: 1600 avg loss: 0.34902535794484124  avg acc: 0.88
iter: 1700 avg loss: 0.6791399506655129  avg acc: 0.81
iter: 1800 avg loss: 0.3951586079548726  avg acc: 0.87
iter: 1900 avg loss: 0.4159683063603881  avg acc: 0.89
iter: 2000 avg loss: 0.27816404083230056  avg acc: 0.92
iter: 2100 avg loss: 0.5605833697203857  avg acc: 0.84
iter: 2200 avg loss: 0.37285155898524813  avg acc: 0.9
iter: 2300 avg loss: 0.41850831626878493  avg acc: 0.85
iter: 2400 avg loss: 0.7936931391355185  avg acc: 0.76
iter: 2500 avg loss: 0.3297072744761994  avg acc: 0.9
iter: 2600 avg loss: 0.34565421371069754  avg acc: 0.87
iter: 2700 avg loss: 0.5018575625178973  avg acc: 0.84
iter: 2800 avg loss: 0.5270823042350417  avg acc: 0.86
iter: 2900 avg loss: 0.48717116257977067  avg acc: 0.89
iter: 3000 avg loss: 0.4052530122955117  avg acc: 0.89
iter: 3100 avg loss: 0.26687815169398593  avg acc: 0.9
iter: 3200 avg loss: 0.3795758736405989  avg acc: 0.87
iter: 3300 avg loss: 0.5796593365946876  avg acc: 0.86
iter: 3400 avg loss: 0.34736460145593023  avg acc: 0.9
iter: 3500 avg loss: 0.4136715810051541  avg acc: 0.86
iter: 3600 avg loss: 0.2703443400157175  avg acc: 0.93
iter: 3700 avg loss: 0.5784640507144941  avg acc: 0.81
iter: 3800 avg loss: 0.20622716832062615  avg acc: 0.95
iter: 3900 avg loss: 0.41141472956035385  avg acc: 0.91
iter: 4000 avg loss: 0.3663108402841854  avg acc: 0.93
iter: 4100 avg loss: 0.38356798157303634  avg acc: 0.9
iter: 4200 avg loss: 0.43840669860597087  avg acc: 0.88
iter: 4300 avg loss: 0.4106599461663881  avg acc: 0.82
iter: 4400 avg loss: 0.2917253330227588  avg acc: 0.9
iter: 4500 avg loss: 0.3546073785807105  avg acc: 0.93
iter: 4600 avg loss: 0.42957394661908116  avg acc: 0.89
iter: 4700 avg loss: 0.3119520882346803  avg acc: 0.93
iter: 4800 avg loss: 0.4555678005747809  avg acc: 0.88
iter: 4900 avg loss: 0.3197443329783045  avg acc: 0.88
iter: 5000 avg loss: 0.532776386032621  avg acc: 0.85
iter: 5100 avg loss: 0.32966807087176675  avg acc: 0.89
iter: 5200 avg loss: 0.260855676336319  avg acc: 0.92
iter: 5300 avg loss: 0.33717387596004605  avg acc: 0.88
iter: 5400 avg loss: 0.6164377640563035  avg acc: 0.85
iter: 5500 avg loss: 0.1496034602282617  avg acc: 0.93
iter: 5600 avg loss: 0.1880811395444831  avg acc: 0.96
iter: 5700 avg loss: 0.3575092199943027  avg acc: 0.91
iter: 5800 avg loss: 0.31796957447712676  avg acc: 0.88
iter: 5900 avg loss: 0.3656350755865679  avg acc: 0.87
iter: 6000 avg loss: 0.32570582138162807  avg acc: 0.92
iter: 6100 avg loss: 0.16242146226074627  avg acc: 0.96
iter: 6200 avg loss: 0.32440848231376535  avg acc: 0.88
iter: 6300 avg loss: 0.354676403040742  avg acc: 0.92
iter: 6400 avg loss: 0.5141892128984139  avg acc: 0.84
iter: 6500 avg loss: 0.4419214200008835  avg acc: 0.83
iter: 6600 avg loss: 0.36014366405938736  avg acc: 0.9
iter: 6700 avg loss: 0.30500876643768515  avg acc: 0.92
iter: 6800 avg loss: 0.39940236237920795  avg acc: 0.91
iter: 6900 avg loss: 0.3054152553775282  avg acc: 0.92
iter: 7000 avg loss: 0.2977108931867772  avg acc: 0.92
iter: 7100 avg loss: 0.2130608581020762  avg acc: 0.91
iter: 7200 avg loss: 0.32642567591462046  avg acc: 0.9
iter: 7300 avg loss: 0.4151272534276844  avg acc: 0.93
iter: 7400 avg loss: 0.27653728071530753  avg acc: 0.91
iter: 7500 avg loss: 0.3619306728513211  avg acc: 0.87
iter: 7600 avg loss: 0.3169471165940523  avg acc: 0.91
iter: 7700 avg loss: 0.28968635383173313  avg acc: 0.9
iter: 7800 avg loss: 0.1536046762082921  avg acc: 0.98
iter: 7900 avg loss: 0.43853786575431164  avg acc: 0.87
iter: 8000 avg loss: 0.4304747142980586  avg acc: 0.85
iter: 8100 avg loss: 0.2703623623682166  avg acc: 0.93
"""