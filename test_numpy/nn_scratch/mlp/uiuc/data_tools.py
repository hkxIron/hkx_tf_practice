"Data Tools"

import numpy as np
import h5py
from cnn.mnist.read_mnist import load_mnist

def one_hot_encoder(labels, dim=10):
    '''
    one hot encoder function
    Input:
        labels  : lables to encode (np.array)
        dims    : one hot lable space dimensions (np.array)
    Outputs:
        one_hot_labels : encoded lables of dims[_, dims](np.array)
    '''
    one_hot_labels = np.zeros((labels.shape[0], dim))
    for i in range(labels.shape[0]):
        one_hot_labels[i][labels[i]] = 1
    return one_hot_labels


def preprocess(data):
    '''
    Normalization
    '''
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (data - data.mean(axis=0)) / data.std(axis=0)
    result[np.isnan(result)] = 0.0
    return result


def loader(data_dir, seed=None):
    '''
    Loader and processor function
    Input:
        data_dir : path to the data to load (str)
        seed     : seed value for shuffle (int)
    return:
        x_train : Processed Training features (np.array)
        y_train : Processed Training lables (np.array)
        x_valid : Processed Validation features (np.array)
        y_valid : Processed Validation lables (np.array)
        x_test  : Processed Testing features (np.array)
        y_test  : Processed Testing lables (np.array)
    '''

    #MNIST_data = h5py.File(data_dir, 'r')

    # Loading test and training
    #mnist_train_data = np.float32(MNIST_data['x_train'][:])
    #mnist_train_labels = np.int32(np.array(MNIST_data['y_train'][:, 0]))

    #mnist_test_data = np.float32(MNIST_data['x_test'][:])
    #mnist_test_labels = np.int32(np.array(MNIST_data['y_test'][:, 0]))
    mnist_train_data, mnist_train_labels = load_mnist("../../cnn/mnist", 'train')
    mnist_test_data, mnist_test_labels = load_mnist("../../cnn/mnist", 'train')


    # Normalization of feature space
    mnist_test_data = preprocess(np.array(mnist_test_data))
    mnist_train_data = preprocess(np.array(mnist_train_data))

    # One hot lable encoding
    mnist_test_labels = one_hot_encoder(np.array(mnist_test_labels))
    mnist_train_labels = one_hot_encoder(np.array(mnist_train_labels))

    # Shuffle data and split.
    if seed is not None:
        np.random.seed(seed)
    n = mnist_train_data.shape[0]
    indices = np.random.permutation(n)
    n_train = int((55.0 / 60) * n)
    train_idx, valid_idx = indices[:n_train], indices[n_train:]

    # Splitting
    x_train, y_train = mnist_train_data[train_idx, :], mnist_train_labels[train_idx, :]
    x_valid, y_valid = mnist_train_data[valid_idx, :], mnist_train_labels[valid_idx, :]
    x_test, y_test = mnist_test_data, mnist_test_labels

    return (x_train, y_train, x_valid, y_valid, x_test, y_test)