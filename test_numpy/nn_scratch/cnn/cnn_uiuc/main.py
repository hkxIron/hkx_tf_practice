"MNIST CNN classifier"

import argparse
import numpy as np
import model.network as nn
from data_tools import loader, accuracy, make_batches

parser = argparse.ArgumentParser(description='MNIST Classification')

# HYPERPARAMETERS
parser.add_argument('-epochs', '--epochs', nargs='?', metavar='',
                    const=50, default=50, type=int, help='Number of training epochs')
parser.add_argument('-batch_size', '--batch_size', nargs='?', metavar='',
                    const=64, default=64, type=int, help='Size of the training batch')
parser.add_argument('-lr', '--lr', nargs='?', metavar='',
                    const=0.02, default=0.01, type=float, help='Learning rate for optimizer')
parser.add_argument('-decay_steps', '--decay_steps', nargs='?', metavar='',
                    const=1000, default=1000, type=int, help='Optimizer decay step')
parser.add_argument('-decay_rate', '--decay_rate', nargs='?', metavar='',
                    const=1.0, default=1.0, type=float, help='Optimizer decay rate')

# SYSTEM PARAMETERS
parser.add_argument('-data_path', '--data_path', nargs='?', metavar='',
                    const='data/MNISTdata.hdf5', default='data/MNISTdata.hdf5', type=str,
                    help='Path to MNIST(.hdf5) data')

args = parser.parse_args()


def test(model, X_test, y_test):
    '''
    Model testing high level pipeline
    Inputs:
        model   : Model (model obj)
        x_valid : Validation features (np.array)
        y_valid : Validation lables (np.array)
    Return:
        acc : accuracy on test set (float)
        loss: loss on test set (float)
    '''
    batches = make_batches(X_test, y_test, args.batch_size)

    tot_loss = 0
    acc = 0

    for X_batch, y_batch in batches:
        loss, _ = model.train_step(X_batch, y_batch)
        acc += accuracy(y_batch, model.predict(X_batch))
        tot_loss += loss

    return acc / len(batches), tot_loss / len(batches)


def train(model, X_train, y_train, X_valid, y_valid):
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

    opt = nn.GradientDescentOptimizer(args.lr, args.decay_steps, args.decay_rate)

    batches = make_batches(X_train, y_train, args.batch_size)

    for i in range(args.epochs):
        tot_loss = 0
        acc = 0

        for X_batch, y_batch in batches:
            loss, _ = model.train_step(X_batch, y_batch)
            tot_loss += loss

            opt.update(model)
            acc += accuracy(y_batch, model.predict(X_batch))

        train_loss = tot_loss / len(batches)

        train_accuracy = acc / len(batches)
        # print(train_accuracy)
        # train_accuracy = accuracy(y_train, model.predict(X_train))

        valid_accuracy, valid_loss = test(model, X_valid, y_valid)

        # Verbose
        msg = "Epoch: {0:>2}, Training Loss: {1:>6.4f}, Training Acc: {2:>6.3%}, Validation Loss: {3:>6.4f}, Validation Acc: {4:>6.3%}"
        print(msg.format(i + 1, train_loss, train_accuracy, valid_loss, valid_accuracy))


def main():
    '''
    High level pipeline for MNIST CNN classifier
    '''
    # Data Loading and Processing
    print('Loading Data...' + '\n')
    training_set, test_set = loader(args.data_path)
    X, y = training_set
    X_test, y_test = test_set

    # Dimensions
    mnist_dims = (1, 28, 28)

    # Model Initialization
    model = nn.CNN(mnist_dims, num_class=10)

    print('-' * 20 + 'Started Training' + '-' * 20 + '\n')
    train(model, X, y, X_test, y_test)


if __name__ == "__main__":
    main()