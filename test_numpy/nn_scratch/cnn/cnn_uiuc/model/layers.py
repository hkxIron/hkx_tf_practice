import numpy as np

class Conv():

    def __init__(self, X_dim, out_channels, K_height, K_width, stride, padding):
        """
        Constructor for Convolution Inititalization

        Input:
            X_dim   : Dimensions of the image (tuple)
            channels: Number of channels for output (int)
            K_height: Height of filter (int)
            K_width : Width of filter (int)
            stride  : stride value (int)
            padding : padding value (int)
        """

        # Stride and padding
        self.stride = stride
        self.padding = padding

        # Kernel dimensions
        self.out_channels = out_channels
        self.K_height = K_height
        self.K_width = K_width

        # Image tensor dimensions
        self.d_X, self.h_X, self.w_X = X_dim # [input_channel, height, width]

        # Init kernel
        # K:[out_channel, input_channel, height, width]
        self.K = np.random.randn(out_channels, self.d_X, K_height, K_width) / np.sqrt(out_channels / 2.)
        self.b = np.zeros((self.out_channels, 1))
        self.params = [self.K, self.b]

        #### DEBUG -->
        self.h_out = (self.h_X - K_height + 2 * padding) / stride + 1
        self.w_out = (self.w_X - K_width + 2 * padding) / stride + 1

        # Output dimensions
        self.h_out, self.w_out = int(self.h_out), int(self.w_out)
        self.out_dim = (self.out_channels, self.h_out, self.w_out)

    def forward(self, X):
        """ Forward propogation """

        # X:[batch, input_dim=1, height=28, width=28]
        # Number of samples cache
        self.n_X = X.shape[0]

        # 将卷积转化为两矩阵相乘
        # receptive field for the images 'X'
        self.X_col = image2field_index(X, self.K_height, self.K_width, stride=self.stride, padding=self.padding)

        # Flat the Kernel matrix
        Flat_K = self.K.reshape(self.out_channels, -1)

        # Receptor field index with kernel multiplication
        out = np.matmul(Flat_K, self.X_col) + self.b

        # Reshaping the output to the calculated output
        out = out.reshape(self.out_channels, self.h_out, self.w_out, self.n_X)
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """ Back propogation """

        # Flatten the derivative matrix
        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        # Calculate the Kernel grad and bias grad
        dK = np.matmul(dout_flat, self.X_col.T)
        dK = dK.reshape(self.K.shape)
        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.out_channels, -1)

        # Flat the kernel
        K_flat = self.K.reshape(self.out_channels, -1)

        # Calulate the grad wrt X (image)
        dX_col = np.matmul(K_flat.T, dout_flat)
        shape = (self.n_X, self.d_X, self.h_X, self.w_X)
        dX = field2image_index(dX_col, shape, self.K_height,
                               self.K_width, self.padding, self.stride)

        return dX, [dK, db]


class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X):
        """ Forward propogation """
        self.X_shape = X.shape
        self.out_shape = (self.X_shape[0], -1)
        out = X.ravel().reshape(self.out_shape)
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        """ Back propogation """
        out = dout.reshape(self.X_shape)
        return out, ()


class FullyConnected():

    def __init__(self, in_size, out_size):
        # W:[in_size, out_size]
        self.W = np.random.randn(in_size, out_size) / np.sqrt(in_size / 2.)
        # b:[1, out_size]
        self.b = np.zeros((1, out_size))
        self.params = [self.W, self.b]

    def forward(self, X):
        """ Forward propogation """
        """
        X:[batch,*,in_size]
        W:[in_size, out_size]
        out:[batch,*,out_size]
        out = X*W+b
        """
        self.X = X
        out = self.X @ self.W + self.b
        return out

    def backward(self, dout):
        """ Back propogation
        :param dout: top层的梯度
        :return: 下一层的梯度, 模型需要更新的参数

        dL/dW =dL/dout*dout/dW
              =X.T*dL/dout
        dout:[batch, *, out_size]
        W:[in_size, out_size]
        X:[batch,*,in_size]
        """
        dW = self.X.T @ dout # 矩阵相乘,注意:在batch维度,梯度是相加的
        db = np.sum(dout, axis=0) # 因此b也要相加
        dX = dout @ self.W.T # 对X的梯度,传入下一层
        return dX, [dW, db]

class ReLU():
    def __init__(self):
        self.params = []

    def forward(self, X):
        """ Forward propogation """
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        """ Back propogation """
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, []


class sigmoid():
    def __init__(self):
        self.params = []

    def forward(self, X):
        """ Forward propogation """
        out = 1.0 / (1.0 + np.exp(X))
        self.out = out
        return out

    def backward(self, dout):
        """ Back propogation """
        dX = dout * self.out * (1 - self.out)
        return dX, []


'''
    Utility functions were provided by Stanford CS 231 and 
    also explained in the UIUC CS446 as receptive field explaination.
    link : https://github.com/ShibiHe/Stanford-CS231n-assignments/blob/master/assignment3/cs231n/im2col.py
    I have used comments to explain the flow of the methods.
'''
def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    # introducing i, j, k
    # ->k is of shape (C field_height field_width, 1)
    # ->i is of shape (C field_height field_width, out_heightout_width)
    # ->j is of shape (C field_height field_width, out_heightout_width)

    # The i1, j1 in functionget_im2col_indices is used to tuned the width
    # and height index positionS according to different neurons

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)

    # Because we also traverse width side first for all neurons in each
    # filter, so we use np.tile for j1(row), np.repeat for i1(column).

    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), int(out_height))
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C, dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def image2field_index(x, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """

    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    # we are extract out a matrix of 3-D (N, C field_height field_width,
    # out_heightout_width), the broadcasted index matrix is of dimension
    # (C field_height field_width, out_heightout_width)
    cols = x_padded[:, k, i, j]

    # Reshaping it to columns indx
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def field2image_index(cols, x_shape, field_height=3, field_width=3, padding=1,
                      stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]