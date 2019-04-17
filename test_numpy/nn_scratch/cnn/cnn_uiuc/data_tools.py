"Data Tools"
import os
import h5py
import gzip
import numpy as np
from cnn.mnist.read_mnist import load_mnist

def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)

def make_batches(X, y, batch_size, shuffle=True, seed=None):
    '''
    Making Batches  from data

    Input:
        X           : Images tensor (np.array)
        y           : Lable array (np.array)
        batch_size  : Size of the batch (int)
        shuffle     : Shuffling Indicator (Bool)
        seed        : random number seed (int)

    Return:
        batches : list containing batches (list)
    '''

    m = X.shape[0]
    batches = []

    # Shuffle
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        n = X.shape[0]
        indices = np.random.permutation(n)
        X, y = X[indices, :, :, :], y[indices]

    # Making batches
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size, :, :, :]
        y_batch = y[i:i + batch_size, ]
        batches.append((X_batch, y_batch))
    return batches


def loader(path, num_training=None, num_test=None):
    '''
    Loader for loading and preprossing

    Input:
        path        : path to the data (str)
        num_training: number of training samples (int)
        num_test    : number of testing samples (int)

    Return:
        Training and Testing sets (tuples)
    '''

    # MNIST_data = h5py.File(path, 'r')

    # Loading test and training
    """
    X_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))

    X_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
    """
    X_train, y_train = load_mnist("../mnist", 'train')
    X_test, y_test = load_mnist("../mnist", 'train')

    shape = (-1, 1, 28, 28)
    X_train = X_train.reshape(shape)
    X_test = X_test.reshape(shape)

    # Subseting if necessary
    if num_training is not None:
        X_train, y_train = X_train[range(num_training)], y_train[range(num_training)]
        X_test, y_test = X_test[range(num_test)], y_test[range(num_test)]

    return (X_train, y_train), (X_test, y_test)