import numpy as np
import random

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.
    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.
    You must implement the optimization in problem 1(a) of the
    written assignment!
    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    # 利用公式实现优化版的softmax，要求既能处理向量，也能处理矩阵（视作多个不相干的行向量集合）。
    orig_shape = x.shape
    if len(x.shape) > 1:
        # Matrix
        ### YOUR CODE HERE
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax, 1, x)  # 沿第1维进行处理，也就是每一行进行处理，即对每行减去行最大值
        denominator = np.apply_along_axis(denom, 1, x) #对每行除以行最大值
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0], 1))
        x = x * denominator
        ### END YOUR CODE
    else:
        # Vector
        ### YOUR CODE HERE
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
        ### END YOUR CODE
    assert x.shape == orig_shape
    return x

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    ### YOUR CODE HERE
    s = 1.0 / (1 + np.exp(-x))
    ### END YOUR CODE
    return s
def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.
    Arguments:
    s -- A scalar or numpy array.
    Return:
    ds -- Your computed gradient.
    """
    ### YOUR CODE HERE
    ds = s * (1 - s)
    ### END YOUR CODE
    return ds

def gradcheck_naive(f, x):
    """ Gradient check for a function f.
    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.
        ### YOUR CODE HERE:
        x[ix] += h
        random.setstate(rndstate)
        new_f1 = f(x)[0]
        x[ix] -= 2*h
        random.setstate(rndstate)
        new_f2 = f(x)[0]
        x[ix] += h
        numgrad = (new_f1 - new_f2) / (2 * h)
        ### END YOUR CODE
        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % ( grad[ix], numgrad))
            return
        it.iternext() # Step to next dimension
    print("Gradient check passed!")

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    ### YOUR CODE HERE: forward propagation
    h = sigmoid(np.dot(data,W1) + b1)
    yhat = softmax(np.dot(h,W2) + b2)
    ### END YOUR CODE
    ### YOUR CODE HERE: backward propagation
    cost = np.sum(-np.log(yhat[labels==1])) / data.shape[0]
    d3 = (yhat - labels) / data.shape[0]
    gradW2 = np.dot(h.T, d3)
    gradb2 = np.sum(d3,0,keepdims=True)
    dh = np.dot(d3,W2.T)
    grad_h = sigmoid_grad(h) * dh
    gradW1 = np.dot(data.T,grad_h)
    gradb1 = np.sum(grad_h,0)
    ### END YOUR CODE
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))
    return cost, grad

def main():
    mat = np.arange(6).reshape((2,3))
    print("mat:",mat)
    mat_softmaxt = softmax(mat)
    print("mat_softmax:",mat_softmaxt)

if __name__ == '__main__':
    main()
