#  https://github.com/hkxIron/deeplearning/blob/master/MP1/model/network.py
"Model"
"""
本代码组织结构与caffe十分类似,写得非常好,十分有利于学习与理解
"""

import numpy as np

# Utility functions.
def linear(x):
    "linear function"
    return x

def d_linear(activate_value=None, x=None):
    "Derivative of linear function"
    return 1.0


def sigmoid(x):
    "The sigmoid function."
    return 1 / (1 + np.exp(-x))

def d_sigmoid(activated_value=None, x=None):
    "Derivative of sigmoid function"
    if activated_value is not None:
        return activated_value*(1-activated_value)
    else:
        return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    "The rectified linear activation function."
    return np.clip(x, a_min=0.0, a_max=None)

def d_relu(activated_value=None, x=None):
    "Derivative of RELU given activation (a) or input (x)."
    # 对于relu, activated与x的导数计算是一样的
    if activated_value is not None:
        d = np.zeros_like(activated_value)
        d[np.where(activated_value > 0.0)] = 1.0
        return d
    else:
        return d_relu(activated_value=relu(x))

def tanh(x):
    "The tanh activation function."
    return np.tanh(x)

def d_tanh(activate_value=None, x=None):
    "The derivative of the tanh function."
    if activate_value is not None:
        return 1 - activate_value ** 2
    else:
        return d_tanh(activate_value=tanh(x))

def softmax(x):
    "Softmax function"
    # For numerical stability mentioned in CS226 UCB
    shifted_x = x - np.max(x, axis=1, keepdims=True) # 以防止溢出

    f = np.exp(shifted_x)
    p = f / np.sum(f, axis=1, keepdims=True)
    return p

# d_softmax比较复杂
# delta(k,k')*y(k)-y(k)*y(k')

def mean_cross_entropy(p, y):
    "Mean cross entropy"
    n = y.shape[0]
    return - np.sum(y * np.log(np.clip(p, a_min=1e-5, a_max=None))) / n

def mean_cross_entropy_softmax(logits, y):
    "Mean cross entropy with the softmax function"
    return mean_cross_entropy(softmax(logits), y)

def d_mean_cross_entropy_softmax(logits, y):
    "derivative of the Error w.r.t Mean cross entropy with the softmax function"
    return softmax(logits) - y

# Mapping from activation functions to its derivatives.
# 用函数名作key来查找对应的导数
fun_to_d_fun = {relu: d_relu,
                tanh: d_tanh,
                sigmoid: d_sigmoid,
                linear: d_linear}

class Layer(object):
    "Implements a layer of a NN."

    def __init__(self, shape, activ_func):

        input_dim = shape[0]
        output_dim = shape[1]
        # Weight matrix of dims[L-1, L]
        # w:[L-1, L]
        self.w = np.random.uniform(-np.sqrt(2.0 / input_dim),
                                   np.sqrt(2.0 / output_dim),
                                   size=shape)

        # Bias marix of dims[1, L]
        self.b = np.zeros((1, output_dim))

        # The activation function
        self.activate_func = activ_func

        # The derivative of the activation function.
        self.d_activate = fun_to_d_fun[activ_func] # 获取导数

    def forward(self, inputs):
        '''
        Forward propagate through this layer.
        Inputs:
            inputs : inputs to this layer of dim[N, L-1] (np.array)
        Outputs:
            outputs: output of this layer of dim[N, L] (np.array)
        '''
        # cache for backward
        self.inputs = inputs

        # Linear score
        # inputs:[batch, inputs=L-1]
        # w:[L-1, L]
        # score:[batch, outputs=L]
        # b:[1, L]
        score = np.dot(inputs, self.w) + self.b
        self.score = score

        # Activation
        # score:[batch, outputs=L]
        outputs = self.activate_func(score)

        # cache for backward
        self.activate_value = outputs # 激活后的输出

        return outputs

    def backward(self, d_outputs):
        """
        Backward propagate the gradient through this layer.
        Inputs:
            d_outputs : deltas from the previous(deeper) layer of dims[N, L] (np.array)
        Outputs:
            d_inputs : deltas for this layer of dims[N, L-1] (np.array)
        """

        # Derivatives of the loss w.r.t the scores (the result from linear transformation).
        # forward:
        # scores = inputs*w+b
        # outputs = activate_value = f(scores)

        # backward:
        # d_scores = dL/d_scores =
        #          = dL/d_activate{i+1}*d_activate{i+1}/d_scores
        #          = dL/d_activate{i+1}*f'(scores)
        #          = d_outputs*f'(scores)
        # d_outputs = dL/d_a{i+1}, shape=[batch, L]
        # d_activate:[batch,L]
        # d_scores:[batch, L]
        # d_scores = d_outputs * self.d_activate(self.score) # 我感觉这里有些问题,这里应该是激活前的值吧

        # Derivatives of the loss w.r.t the bias, averaged over all data points.
        # 注意,对于b而言,此处是batch里的平均值
        """
        原始是激活后的值,
        d_scores = d_outputs * self.d_activate(activate_value=self.activate_value)
        或者用激活前的值,可与公式保持一致,如下:
        """
        d_scores = d_outputs * self.d_activate(x=self.score)

        # dL/d_scores:[batch, L]
        # d_b = dL/db = dL/d_scores*d_scores/db = dL/d_scores*1
        # d_b:[1, L], 必须与batch无关,因此平均
        self.d_b = np.mean(d_scores, axis=0, keepdims=True)

        # Derivatives of the loss w.r.t the weight matrix, averaged over all data points.
        # scores = input*w
        # inputs:[batch, L-1]
        # d_scores:[batch, L]
        # d_w:[L-1, L], dL/dw = dL/d_score*d_scores/dw =dL/d_score* inputs
        self.d_w = np.dot(self.inputs.T, d_scores) / d_scores.shape[0] # 此处需要除以batch_size,以防止w的更新与batch大小有关

        # Derivatives of the loss w.r.t the previous layer's activations/outputs.
        # d_inputs = dL/d_inputs = dL/d_scores *d_scores/d_inputs
        #          = dL/d_scores*w
        # d_scores:[batch, L]
        # w:[L-1, L]
        # d_inputs: [batch, L-1]
        d_inputs = np.dot(d_scores, self.w.T)

        return d_inputs  # 对x的梯度,将会传入下一层中

# 多层感知机, MLP
class Perceptron(object):
    "Perceptron Model"

    def __init__(self, input_dim, output_dim, hidden_dims, activ_funcs):
        """
        Input:
          input_dim     : dimension of input (int)
          output_dim    : dimension of output( int)
          hidden_dims    : a list of integers specifying the number of
                            hidden units on each layer.
          activ_funcs   : a list of function objects specifying
                            the activation function of each layer.
        """

        self.units = [input_dim] + hidden_dims[:] + [output_dim]
        self.activ_funcs = activ_funcs[:] + [linear] # 最后一层需要加上线性层,其实是为了凑数,直接返回的

        self.shapes = [] # 各层的shape num
        self.layers = []

        self.logits = None
        self.pred_prob = None

        self._build_network(self.units)

    def _build_network(self, neuron_list):
        """
        Build the network by assigning tuples to shapes and
        objects to layers.
        Input:
          neuron_list     : list of units in each layers (list)
        """

        # num_input and num_outputs assignment per layer
        for i in range(len(neuron_list) - 1):
            self.shapes.append((neuron_list[i], neuron_list[i + 1]))

        # creating layers
        for i, shape in enumerate(self.shapes):
            self.layers.append(Layer(shape, self.activ_funcs[i]))

    def loss(self, outputs, labels):
        """
        Compute the cross entropy softmax loss
        Inputs:
            outputs : output of the last layer
            gt      : ground truth lables
        Returns:
            mean_cross_entropy_softmax
        """
        return mean_cross_entropy_softmax(outputs, labels)

    def d_loss(self, outputs, labels):
        """
        Compute derivatives of the cross entropy softmax loss w.r.t the outputs.
        Inputs:
            outputs : output of the last layer
            labels      : ground truth lables
        Returns:
            derivative w.r.t mean_cross_entropy_softmax
        """
        return d_mean_cross_entropy_softmax(outputs, labels)

    def forward(self, x, labels=None):
        """
        Network inference / forward propogation.
        Inputs:
            x  : features of dim[batch_size, feature_dims] (np.array)
            labels : lables of dim[batch_size, lable_dims] (np.array) [optional]
        Returns:
            pred_prob : prediction probability of dim[batch_size, lable_dims] (np.array)
            loss      : loss value (float) [if gt is None then None]
        """
        layer_inputs = x
        # 逐层向前传播
        for layer in self.layers:
            layer_outputs = layer.forward(layer_inputs)
            layer_inputs = layer_outputs

        self.logits = layer_outputs

        self.pred_prob = softmax(self.logits)

        if labels is None:
            return self.pred_prob, None
        else:
            return self.pred_prob, self.loss(layer_outputs, labels)

    def backward(self, labels):
        """
        Network train / back propogation.
        Inputs:
            labels : lables of dim[batch_size, lable_dims] (np.array) [optional]
        """
        # 注意:此处提取的梯度为loss对最后一层激活后的值(logits)的梯度,
        # L = loss(label, y), y = sigmoid(a),dL/da = dL/dy*dy/da,
        # 此处 d_layer_outputs = dL/da, 即所谓的delta
        d_layer_outputs = self.d_loss(self.layers[-1].activate_value, labels)
        # 逐层反向传播
        for layer in self.layers[::-1]:
            d_layer_inputs = layer.backward(d_layer_outputs)
            d_layer_outputs = d_layer_inputs

    def predict(self, x):
        '''
        Network predition
        Inputs:
            x  : features of dim[batch_size, feature_dims] (np.array)
        Outputs:
            y_predict : one hot predition of dim[batch_size, label_dims] (np.array)
        '''
        pred_prob, _ = self.forward(x)
        return np.argmax(pred_prob, axis=1)



class GradientDescentOptimizer(object):
    "Gradient descent with staircase exponential decay."

    def __init__(self, learning_rate, decay_steps=1000,
                 decay_rate=1.0):

        self.learning_rate = learning_rate
        self.steps = 0.0
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def update(self, model):
        '''
        Update model parameters.
        Inputs:
            model : Model (model obj)

        更新模型的参数
        '''
        for layer in model.layers:
            layer.w -= layer.d_w * self.learning_rate
            layer.b -= layer.d_b * self.learning_rate
        self.steps += 1
        if (self.steps + 1) % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate