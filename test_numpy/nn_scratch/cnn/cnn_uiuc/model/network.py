import numpy as np
from model.layers import *


# Utility function for the network

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def mean_cross_entropy_with_softmax(model_out, y_gt):
    m = y_gt.shape[0]
    p = softmax(model_out)
    log_likelihood = -np.log(p[range(m), y_gt])
    loss = np.sum(log_likelihood) / m

    dx = p.copy()
    dx[range(m), y_gt] -= 1
    dx /= m
    return loss, dx


def l2_regularization(layers, lam=0.001):
    reg_loss = 0.0
    for layer in layers:
        if hasattr(layer, 'W'):
            reg_loss += 0.5 * lam * np.sum(layer.W * layer.W)
    return reg_loss


def delta_l2_regularization(layers, grads, lam=0.001):
    for layer, grad in zip(layers, reversed(grads)):
        if hasattr(layer, 'W'):
            grad[0] += lam * layer.W
    return grads


class CNN:
    """ Convolution Neural Net model"""

    def __init__(self, X_dim, num_class,
                 loss_func=mean_cross_entropy_with_softmax):

        # Builds the model and save the components
        self.layers = self._build_network(X_dim, num_class)
        self.params = []

        # Cache the parameters
        for layer in self.layers:
            self.params.append(layer.params)

        # Loss function
        self.loss_func = loss_func

    def _build_network(self, X_dim, num_class):

        # 1. Convolution layer
        # 2. Relu
        # 3. Flat the output
        # 4. Fullyconnected layer

        conv = Conv(X_dim,
                    channels=32,
                    K_height=7,
                    K_width=7,
                    stride=1,
                    padding=1)

        relu_conv = ReLU()

        flat = Flatten()

        fc = FullyConnected(np.prod(conv.out_dim), num_class)

        return [conv, relu_conv, flat, fc]

    def forward(self, X):
        """ Forward propogation """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        """ Back propogation """
        grads = []
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, X, y):

        # Forward (Inference)
        out = self.forward(X)

        # Loss and grad calc
        loss, dout = self.loss_func(out, y)
        loss += l2_regularization(self.layers)
        grads = self.backward(dout)
        grads = delta_l2_regularization(self.layers, grads)

        # cache
        self.loss, self.grads = loss, grads
        return loss, grads

    def predict(self, X):
        """ Prediction """

        X = self.forward(X)
        return np.argmax(softmax(X), axis=1)


class GradientDescentOptimizer(object):
    "Gradient descent with staircase exponential decay."

    def __init__(self, learning_rate, decay_steps=1000,
                 decay_rate=1.0):

        self.learning_rate = learning_rate
        self.steps = 0.0
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def update(self, model):

        for param, grad in zip(model.params, reversed(model.grads)):
            for i in range(len(grad)):
                param[i] += - self.learning_rate * grad[i]

        self.steps += 1
        if (self.steps + 1) % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate