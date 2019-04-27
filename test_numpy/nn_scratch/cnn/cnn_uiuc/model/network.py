import numpy as np
from model.layers import *

# Utility function for the network
def softmax(x):
    # x:[batch, num_class]
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def mean_cross_entropy_with_softmax(model_out, y_gt):
    # model_out:[batch, num_class]
    # y:[batch]
    m = y_gt.shape[0] # m:batch
    # p:[batch, num_class]
    p = softmax(model_out)
    # p:[batch, num_class]
    # loss: - sum_i{ yi*log(pi) } = - sum_i{ log(pi) } where yi=1
    # 由于yi = {0,1}, yi为0时,对于loss无影响,因此只需要取yi=1的元素
    negative_log_likelihood = -np.log(p[range(m), y_gt])
    loss = np.sum(negative_log_likelihood) / m

    dx = p.copy()
    # dx = d model_out = p - y
    # 由于yi = {0,1}, yi=0时, dx = p, yi=1时, dx = p - 1
    dx[range(m), y_gt] -= 1
    dx /= m
    # loss:scalar
    # dx:[batch, num_class]
    return loss, dx

def l2_regularization(layers, reversed_grads, lam=0.001):
    # reversed_grads: 按层逆序后的梯度
    reg_loss = 0.0
    for layer, grad in zip(layers, reversed(reversed_grads)):
        if hasattr(layer, 'W'):
            reg_loss += 0.5 * lam * np.sum(layer.W * layer.W)
            grad[0] += lam * layer.W # grad = [dw, db], bias不需要梯度
    return reg_loss, reversed_grads

class CNN:
    """ Convolution Neural Net model"""

    def __init__(self,
                 X_dim,
                 num_class,
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

        # X:[batch, channel=input_dim, height=28, width=28]
        # conv:[batch, output_channel, height - K_height+2*pad+1, width-K_width+2*pad+1]
        conv = Conv(X_dim=X_dim,
                    out_channels=32, # output channel
                    K_height=7,
                    K_width=7,
                    stride=1,
                    padding=1)

        # relu_conv:[batch, output_channel, height - K_height+1, width-K_width+1]
        relu_conv = ReLU()

        # flat:[batch, output_channel*(height - K_height+1)*(width-K_width+1)]
        flat = Flatten()

        # fc:[batch, num_class]
        fc = FullyConnected(in_size=np.prod(conv.out_dim), out_size=num_class)

        return [conv, relu_conv, flat, fc]

    def forward(self, X):
        """ Forward propogation """
        # X:[batch, channel, height=28, width=28]
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        """ Back propogation """
        grads = [] # 后面层的梯度放在前面
        """
        dout是对该层输入的x的梯度,grad是对参数w的梯度
        """
        for layer in reversed(self.layers):
            dout, grad = layer.backward(dout)
            grads.append(grad)
        return grads

    def train_step(self, X, y):

        # Forward (Inference)
        # X:[batch, channel, height, width]
        # y:[batch]
        # out:[batch, num_class]
        out = self.forward(X)
        # dout:[batch, num_class]
        # loss:scalar
        loss, dout = self.loss_func(out, y)
        # grads: 后面层的梯度放在前面
        grads = self.backward(dout)
        reg_loss, grads = l2_regularization(self.layers, grads)
        #grads = delta_l2_regularization(self.layers, grads)
        loss += reg_loss

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