# coding:utf-8
import os
import struct
import numpy as np
# blog:https://www.cnblogs.com/xianhan/p/9145966.html
"""
MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST). 
训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 
50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据.


load_mnist 函数返回两个数组, 第一个是一个 n x m 维的 NumPy array(images), 这里的 n 是样本数(行数), m 是特征数(列数). 训练数据集包含 60,000 个样本, 测试数据集包含 10,000 样本. 在 MNIST 数据集中的每张图片由 28 x 28 个像素点构成, 每个像素点用一个灰度值表示. 在这里, 我们将 28 x 28 的像素展开为一个一维的行向量, 这些行向量就是图片数组里的行(每行 784 个值, 或者说每行就是代表了一张图片). load_mnist 函数返回的第二个数组(labels) 包含了相应的目标变量, 也就是手写数字的类标签(整数 0-9).


TRAINING SET LABEL FILE (train-labels-idx1-ubyte):

[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.


TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

"""

def load_mnist(path:str, type='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % type)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % type)
    """
    通过使用上面两行代码, 我们首先读入 magic number, 它是一个文件协议的描述, 
    也是在我们调用 fromfile 方法将字节读入 NumPy array 之前在文件缓冲中的 item 数(n). 
    作为参数值传入 struct.unpack 的 >II 有两个部分:

    >: 这是指大端(用来定义字节是如何存储的);
    I: 这是指一个无符号整数(4个字节).
    """
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8)) # 大端存储,读取8个字节
        print("type:",type ,"label magic:",magic ,"label number:", n)
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16)) # 大端存储,读取16个字节
        print("type:",type, "image magic:", magic, "image number:", num)
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = np.float32(images)/255.0 # 除以255作归一化

    return images, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    images, labels = load_mnist(".", "t10k")

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )

    ax = ax.flatten()
    for i in range(10):
        #img = images[i].reshape(28, 28)
        img = images[labels == i][0].reshape(28, 28) # 每个样例下只取一个
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

"""
np.savetxt('train_img.csv', X_train,
           fmt='%i', delimiter=',')
np.savetxt('train_labels.csv', y_train,
           fmt='%i', delimiter=',')
np.savetxt('test_img.csv', X_test,
           fmt='%i', delimiter=',')
np.savetxt('test_labels.csv', y_test,
           fmt='%i', delimiter=',')
"""