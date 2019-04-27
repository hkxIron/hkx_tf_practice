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
        # K:[out_channel, input_channel, k_height, k_width]
        self.K = np.random.randn(out_channels, self.d_X, K_height, K_width) / np.sqrt(out_channels / 2.)
        # b:[out_channel, 1]
        self.b = np.zeros((self.out_channels, 1))
        self.params = [self.K, self.b]

        #### DEBUG -->
        self.h_out = (self.h_X - K_height + 2 * padding) / stride + 1
        self.w_out = (self.w_X - K_width + 2 * padding) / stride + 1

        # Output dimensions
        self.h_out, self.w_out = int(self.h_out), int(self.w_out) # 整除
        self.out_dim = (self.out_channels, self.h_out, self.w_out)

    def forward(self, X):
        """ Forward propogation """

        # X:[batch, input_channel=1, height=28, width=28]
        # Number of samples cache
        self.n_X = X.shape[0]

        # 将卷积转化为两矩阵相乘,注意:此处的重点是对X进行重排,而不是对kernel重排
        # receptive field for the images 'X'
        # X_col:[input_channel*field_height*field_width, out_height*out_width*batch]
        #   即 =>:[input_channel*K_height*K_width, out_height*out_width*batch]
        self.X_col = image2field_index(X, self.K_height, self.K_width, stride=self.stride, padding=self.padding)

        # Flat the Kernel matrix
        # K:[out_channel, input_channel, k_height, k_width]
        # Flat_K:[out_channel, input_channel*k_height*k_width]
        Flat_K = self.K.reshape(self.out_channels, -1)

        # Receptor field index with kernel multiplication
        # Flat_K:[out_channel, input_channel*K_height*K_width]
        # out: [out_channel, out_height*out_width*batch]
        out = np.matmul(Flat_K, self.X_col) + self.b

        # Reshaping the output to the calculated output
        # out: [out_channel, out_height, out_width, batch]
        out = out.reshape(self.out_channels, self.h_out, self.w_out, self.n_X)
        # out: [batch, out_channel, out_height, out_width]
        out = out.transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        """ Back propogation """

        # Flatten the derivative matrix
        # dout: [batch, out_channel, out_height, out_width]
        # => [out_channel, out_height, out_width, batch]
        # dout_flat=>[out_channel, out_height*out_width*batch]
        dout_flat = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        """
        out = flat_k * X_col
        d_(flat_k) = d_out* X_col.T
        """
        # Calculate the Kernel grad and bias grad
        # dout_flat:[out_channel, out_height*out_width*batch]
        # X_col:[input_channel*field_height*field_width, out_height*out_width*batch]
        # X_col.T:[out_height*out_width*batch, input_channel*field_height*field_width]
        # dK:[out_channel, input_channel*field_height*field_width]
        dK = np.matmul(dout_flat, self.X_col.T)
        # dK:[out_channel, input_channel, k_height=field_height, k_width=field_width]
        dK = dK.reshape(self.K.shape)
        # dout: [batch, out_channel, out_height, out_width]
        #    np.sum=>[1,out_channel,1,1]
        #    reshape=>[out_channel,1]
        # db:[out_channel,1]
        db = np.sum(dout, axis=(0, 2, 3)).reshape(self.out_channels, -1)

        # Flat the kernel
        # K:[out_channel, input_channel, k_height, k_width]
        # K_flat:[out_channel, input_channel*k_height*k_width]
        K_flat = self.K.reshape(self.out_channels, -1)

        # Calulate the grad wrt X (image)
        # K_flat.T:[input_channel*k_height*k_width, out_channel]
        # dout_flat=>[out_channel, out_height*out_width*batch]
        # dX_col:[input_channel*k_height*k_width, out_height*out_width*batch]
        dX_col = np.matmul(K_flat.T, dout_flat)
        shape = (self.n_X, self.d_X, self.h_X, self.w_X) # [batch, input_channel, height, width]
        # dX:[batch, input_channel, height, width]
        dX = field2image_index(dX_col,
                               shape,
                               self.K_height,
                               self.K_width,
                               self.padding,
                               self.stride)

        return dX, [dK, db]


class Flatten():

    def __init__(self):
        self.params = []

    def forward(self, X):
        """ Forward propogation """
        # X:[batch, input_dim, height, width]
        self.X_shape = X.shape
        # out_shape:[batch, input_dim*height*width]
        self.out_shape = (self.X_shape[0], -1)
        # np.ravel()
        out = X.ravel().reshape(self.out_shape)
        # out_shape:[batch, input_dim*height*width]
        self.out_shape = self.out_shape[1]
        return out

    def backward(self, dout):
        """ Back propogation """
        # out_shape:[batch, input_dim*height*width]
        # out:[batch, input_dim, height, width]
        out = dout.reshape(self.X_shape)
        return out, () # 本身没有参数


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
        X:[batch,in_size]
        W:[in_size, out_size]
        out:[batch,out_size]
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
        dout:[batch, out_size]
        W:[in_size, out_size]
        X:[batch, in_size]
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
        # X:[batch, input_dim, height, width]
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        """ Back propogation """
        dX = dout.copy()
        dX[self.X <= 0] = 0
        return dX, [] # relu没有w参数更新,因此为空

class sigmoid():
    def __init__(self):
        self.params = []

    def forward(self, X):
        """ Forward propogation """
        # X:[batch, input_dim, height, width]
        out = 1.0 / (1.0 + np.exp(X))
        self.out = out
        return out

    def backward(self, dout):
        """ Back propogation """
        """
        out=f(X) = sigmoid(x)
        dL/dX = dL/dout* dout/dX
              = dL/dout* f(x)*(1-f(x))
        """
        dX = dout * self.out * (1 - self.out)
        return dX, [] # sigmoid本身没有参数需要更新

'''
    Utility functions were provided by Stanford CS 231 and 
    also explained in the UIUC CS446 as receptive field explaination.
    link : https://github.com/ShibiHe/Stanford-CS231n-assignments/blob/master/assignment3/cs231n/im2col.py
    I have used comments to explain the flow of the methods.
    
    获取重排X矩阵时对应的下标元素, 分别包括: (channel,  height, width)
'''
def get_im2col_indices(x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    # introducing i, j, k
    # ->k is of shape (C*field_height*field_width, 1)
    # ->i is of shape (C*field_height*field_width, out_height*out_width)
    # ->j is of shape (C*field_height*field_width, out_height*out_width)

    # The i1, j1 in function get_im2col_indices is used to tuned the width
    # and height index position S according to different neurons

    i0 = np.repeat(np.arange(field_height, dtype='int32'), field_width)

    # Because we also traverse width side first for all neurons in each
    # filter, so we use np.tile for j1(column), np.repeat for i1(row).
    i0 = np.tile(A=i0, reps=C)
    i1 = stride * np.repeat(np.arange(out_height, dtype='int32'), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width, dtype='int32'), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1) # x的行下标
    j = j0.reshape(-1, 1) + j1.reshape(1, -1) # x的列下标

    k = np.repeat(np.arange(C, dtype='int32'), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


"""
注意：
此处的field_height, field_width分别针对kernel而言

将X重排成矩阵,以适应kernel,这样的话,重排后的矩阵为dense而非spare矩阵
"""
def image2field_index(x, field_height=3, field_width=3, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """

    # Zero-pad the input
    # x:[batch, input_channel=1, height=28, width=28]
    x_padded = np.pad(array=x,
                      pad_width=((0, 0), # 对于batch axis插入before, after padding
                                 (0, 0), # 对于input_dim axis插入before, after padding
                                 (padding, padding), #对于 height axis插入before, after padding
                                 (padding, padding)),
                      mode='constant')

    # x:[batch, input_channel=1, height=28, width=28]
    # k:[input_channel*field_height*field_width, 1]
    # i:[input_channel*field_height*field_width, out_height*out_width]
    # j:[input_channel*field_height*field_width, out_height*out_width]
    k, i, j = get_im2col_indices(x.shape,
                                 field_height,
                                 field_width,
                                 padding,
                                 stride)

    # we are extract out a matrix of 3-D (N, C*field_height*field_width,out_height*out_width),
    # the broadcasted index matrix is of dimension (C*field_height*field_width, out_height*out_width)
    # cols:[batch, input_channel*field_height*field_width, out_height*out_width]
    cols = x_padded[:, k, i, j]

    # Reshaping it to columns indx
    C = x.shape[1]
    # np.transpose()
    # cols: 交换不同的index:1,2,0
    # cols:[batch, input_channel*field_height*field_width, out_height*out_width]
    #   =>transpose:[input_channel*field_height*field_width, out_height*out_width, batch]
    #   =>  reshape:[input_channel*field_height*field_width, out_height*out_width* batch]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def field2image_index(cols, x_shape, field_height=3, field_width=3, padding=1,
                      stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """

    # 注意:在此处, cols为dL/dout,即后层网络对此层的梯度,需要求梯度dL/dx, dL/dK
    # cols:[input_channel*k_height*k_width, out_height*out_width*batch]
    # x_shape:[batch, input_channel, height, width]
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    # x_shape:[batch, input_channel=1, height=28, width=28]
    # k:[input_channel*field_height*field_width, 1]
    # i:[input_channel*field_height*field_width, out_height*out_width]
    # j:[input_channel*field_height*field_width, out_height*out_width]
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    # cols:[input_channel*k_height*k_width, out_height*out_width*batch]
    # cols_reshaped:[input_channel*k_height*k_width, out_height*out_width, batch]
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    # cols_reshaped:[input_channel*k_height*k_width, out_height*out_width, batch]
    #            => [batch, input_channel*k_height*k_width, out_height*out_width]
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    # x_padded:[batch, input_channel, height, width]
    # cols_reshaped:[batch, input_channel*k_height*k_width, out_height*out_width]
    # indices:[batch, input_channel*k_height*k_width, out_height*out_width], 正好与cols_reshaped的维度一致
    np.add.at(a=x_padded, indices=(slice(None), k, i, j), b=cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

if __name__ == "__main__":
    np.random.seed(0)
    N, C, H, W = 1, 1, 3, 3

    X = np.arange(np.prod([N, C, H, W])).reshape((N, C, H, W))
    k, i, j = get_im2col_indices((N, C, H, W), field_height=2, field_width=2, padding=0, stride=1)
    print("indices_k shape:\n", k.shape) # [4, 1]
    print("indices_i shape:\n", i.shape) # [4, 4]
    print("indices_k:\n", k)
    print("indices_i:\n", i)
    print("indices_j:\n", j)

    print("-------")
    # 将图像重排以适应 kernel
    indexs = image2field_index(X, field_height=2, field_width=2, padding=0, stride=1)
    print("index:\n", indexs)
    print("index shape:\n",indexs.shape) # (4,4),理论上说是一个(4,9)列的重排矩阵
