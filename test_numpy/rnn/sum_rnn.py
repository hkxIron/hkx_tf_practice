# 使用rnn 实现加法
#coding:utf-8
import copy, numpy as np
np.random.seed(0) #固定随机数生成器的种子，便于得到固定的输出，【译者注：完全是为了方便调试用的]
# compute sigmoid nonlinearity
def sigmoid(x): #激活函数
    output = 1/(1+np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):#激活函数的导数
    return output*(1-output)
# training dataset generation
int2binary = {} #整数到其二进制表示的映射
binary_dim = 8 #暂时制作256以内的加法， 可以调大
## 以下5行代码计算0-256的二进制表示
largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
# input variables
alpha = 0.1 #学习速率
input_dim = 2 #因为我们是做两个数相加，每次会喂给神经网络两个bit，所以输入的维度是2
hidden_dim = 16 #隐藏层的神经元节点数，远比理论值要大（译者注：理论上而言，应该一个节点就可以记住有无进位了，但我试了发现4的时候都没法收敛），你可以自己调整这个数，看看调大了是容易更快地收敛还是更慢
output_dim = 1 #我们的输出是一个数，所以维度为1
# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1 #输入层到隐藏层的转化矩阵，维度为2*16， 2是输入维度，16是隐藏层维度
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1
# 译者注：np.random.random产生的是[0,1)的随机数，2 * [0, 1) - 1 => [-1, 1)，
# 是为了有正有负更快地收敛，这涉及到如何初始化参数的问题，通常来说都是靠“经验”或者说“启发式规则”，说得直白一点就是“蒙的”！机器学习里面，超参数的选择，大部分都是这种情况，哈哈。。。
# 我自己试了一下用【0, 2)之间的随机数，貌似不能收敛，用[0,1)就可以，呵呵。。。
# 以下三个分别对应三个矩阵的变化
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
# training logic
# 学习10000个例子
for j in range(100000):
    
    # 下面6行代码，随机产生两个0-128的数字，并查出他们的二进制表示。为了避免相加之和超过256，这里选择两个0-128的数字
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding
    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding
    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    # 存储神经网络的预测值
    d = np.zeros_like(c)
    overallError = 0 #每次把总误差清零
    
    layer_2_deltas = list() #存储每个时间点输出层的误差
    layer_1_values = list() #存储每个时间点隐藏层的值
    layer_1_values.append(np.zeros(hidden_dim)) #一开始没有隐藏层，所以里面都是0
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):#循环遍历每一个二进制位
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])#从右到左，每次去两个输入数字的一个bit位
        y = np.array([[c[binary_dim - position - 1]]]).T#正确答案
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))#（输入层 + 之前的隐藏层） -> 新的隐藏层，这是体现循环神经网络的最核心的地方！！！
        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1)) #隐藏层 * 隐藏层到输出层的转化矩阵synapse_1 -> 输出层
        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2 #预测误差是多少
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2)) #我们把每一个时间点的误差导数都记录下来
        overallError += np.abs(layer_2_error[0])#总误差
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0]) #记录下每一个预测bit位
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))#记录下隐藏层的值，在下一个时间点用
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    #前面代码我们完成了所有时间点的正向传播以及计算最后一层的误差，现在我们要做的是反向传播，从最后一个时间点到第一个时间点
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]]) #最后一次的两个输入
        layer_1 = layer_1_values[-position-1] #当前时间点的隐藏层
        prev_layer_1 = layer_1_values[-position-2] #前一个时间点的隐藏层
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1] #当前时间点输出层导数
        # error at hidden layer
        # 通过后一个时间点（因为是反向传播）的隐藏层误差和当前时间点的输出层误差，计算当前时间点的隐藏层误差
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        # 我们已经完成了当前时间点的反向传播误差计算， 可以构建更新矩阵了。但是我们并不会现在就更新权重矩阵，因为我们还要用他们计算前一个时间点的更新矩阵呢。
        # 所以要等到我们完成了所有反向传播误差计算， 才会真正的去更新权重矩阵，我们暂时把更新矩阵存起来。
        # 可以看这里了解更多关于反向传播的知识
        # http://iamtrask.github.io/2015/07/12/basic-python-network/
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    
    # 我们已经完成了所有的反向传播，可以更新几个转换矩阵了。并把更新矩阵变量清零
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
