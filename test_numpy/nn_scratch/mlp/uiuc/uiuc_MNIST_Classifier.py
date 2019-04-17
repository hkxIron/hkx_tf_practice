# https://github.com/hkxIron/deeplearning/blob/master/MP1/MNIST_Classifier.py
"MNIST classifier"
"""
MP1 : MNIST Classification using perceptron (from scratch)
Objective : Implement and train a neural network from scratch in Python for the MNIST dataset (no PyTorch). The neural network should be trained on the Training Set using stochastic gradient descent. It should achieve 97-98% accuracy on the Test Set. For full credit, submit via Compass (1) the code and (2) a paragraph (in a PDF document) which states the Test Accuracy and briefly describes the implementation.

Implementation : The model I have designed is inspired from Tensorflows Layes API (specifically, fullyconnected API, Optimizer API and activationtion functions). I have written the code which is generalized for any number of hidden layers and activation function. The following is the overall layout of the model :

The entire pipeline is defined in the three layer of abstraction : uiuc_mnist_classifier.py ​(main), data_tools.py (helper module for main) and Network. Main handles the high ​level pipeline of the loading data, training and testing the model. Network is the module consisting of 3 classes and utility functions. Classes are : Perceptron (class to build network), Layers (Class to build one layer i.e. Hidden layers) and StochasticGradientDescent (Class to update weights and biases through optimizer). Utility functions are loss functions, derivative of loss functions, activation functions and their derivatives.

Network 1: Double Hidden Layers : 128 units, 32 units Activations : Relu, Relu
"""

import argparse
import numpy as np
import model.network as nn
from data_tools import loader


parser = argparse.ArgumentParser(description='MNIST Classification')

# HYPERPARAMETERS
parser.add_argument('-epochs', '--epochs', nargs='?', metavar='',
                    const=50, default=50, type=int, help='Number of training epochs')
parser.add_argument('-batch_size', '--batch_size', nargs='?', metavar='',
                    const=100, default=100, type=int, help='Size of the training batch')
parser.add_argument('-lr', '--lr', nargs='?', metavar='',
                    const=0.02, default=0.01, type=float, help='Learning rate for optimizer')

# SYSTEM PARAMETERS
parser.add_argument('-data_path', '--data_path', nargs='?', metavar='',
                    const='data/MNISTdata.hdf5', default='data/MNISTdata.hdf5', type=str, help='Path to MNIST(.hdf5) data')

args = parser.parse_args()


def accuracy(y_predict, y_gt):
    '''helper function for accuracy'''
    return (np.sum(y_predict == y_gt) / np.float(y_gt.shape[0]))


def test(model, x_test, y_test):
    '''
    Model testing high level pipeline
    Inputs:
        model  : Model (model obj)
        x_test : Testing features (np.array)
        y_test : Training lables (np.array)
    '''

    # Inference / forwardprop
    _, test_loss = model.forward(x_test, y_test)
    test_accuracy = accuracy(model.predict(x_test), np.argmax(y_test, axis=1))

    # Verbose
    msg = "Testing Loss: {0:>6.4f}, Testing Acc: {1:>6.3%}"
    print(msg.format(test_loss, test_accuracy))


def train(model, x_train, y_train, x_valid, y_valid):
    '''
    Model training high level pipeline
    Inputs:
        model   : Model (model obj)
        x_train : Training features (np.array)
        y_train : Training lables (np.array)
        x_valid : Validation features (np.array)
        y_valid : Validation lables (np.array)
    Return:
        model : Trained Model (model obj)
    '''

    # Optimizer init
    optimizer = nn.GradientDescentOptimizer(args.lr)

    n_train = x_train.shape[0]

    for i in range(args.epochs):
        tot_loss = 0.0

        for j in range((n_train - 1) // args.batch_size + 1):

            # Creating batches
            batch_start_indx = j * args.batch_size
            batch_end_indx = (j + 1) * args.batch_size
            x_batch = x_train[batch_start_indx:batch_end_indx]
            y_batch = y_train[batch_start_indx:batch_end_indx]

            # Inference / Forwardprop
            _, loss = model.forward(x_batch, y_batch)

            # Total loss
            tot_loss += loss

            # Training / Backprop
            model.backward(labels=y_batch)

            # Model updation
            optimizer.update(model)

        train_loss = tot_loss / (j + 1)

        # Validation Inference / Forwardprop
        print("validation ...")
        _, valid_loss = model.forward(x_valid, y_valid)

        # Accuracy
        train_accuracy = accuracy(model.predict(x_train), np.argmax(y_train, axis=1))
        valid_accuracy = accuracy(model.predict(x_valid), np.argmax(y_valid, axis=1))

        # Verbose
        msg = "Epoch: {0:>2}, Training Loss: {1:>6.4f}, Training Acc: {2:>6.3%}, Validation Loss: {3:>6.4f}, Validation Acc: {4:>6.3%}"
        print(msg.format(i + 1, train_loss, train_accuracy, valid_loss, valid_accuracy))

    return model


def main():
    '''
    High level pipeline for MNIST classifier
    '''

    # Data Loading and Processing
    print('Loading Data...' + '\n')
    x_train, y_train, x_valid, y_valid, x_test, y_test = loader(args.data_path)

    # Dimensions
    features_dims = x_train.shape[1]
    labels_dims = y_train.shape[1]

    # Model Initialization
    # input_dim, output_dim, hidden_dims, activ_funcs
    model = nn.Perceptron(input_dim=features_dims,
                          output_dim=labels_dims,
                          hidden_dims=[128, 32],
                          #activ_funcs=[nn.relu, nn.relu]
                          #activ_funcs=[nn.sigmoid, nn.sigmoid]
                          activ_funcs=[nn.tanh, nn.tanh]
                          ) # 直接将函数对象作为参数传进去

    # Model Training
    print('-' * 20 + 'Started Training' + '-' * 20 + '\n')
    model = train(model, x_train, y_train, x_valid, y_valid)
    print('-' * 20 + 'Training finished' + '-' * 20 + '\n')

    # Model Testing
    print('-' * 20 + 'Started Testing' + '-' * 20 + '\n')
    test(model, x_test, y_test)
    print('-' * 20 + 'Testing finished' + '-' * 20 + '\n')


if __name__ == '__main__':
    main()
