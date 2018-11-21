# blog:http://nicodjimenez.github.io/2014/08/08/lstm.html
# 作者的实现有点类似于caffe,对于第t层,数据从底部输入,从顶部输出,梯度从顶部回传到底部,只不过这里的t是时间time_step

import random

import numpy as np
import math


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return values * (1 - values) # sig(x)*(1-sig(x))


def tanh_derivative(values):
    return 1. - values ** 2  # (1-tanh(x))(1+tanh(x))


# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct # x_dim + hidden_dim
        # weight matrices, 权重不宜初始化为0,而要随机初始化
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) # hidden_dim*(x_dim+hidden_dim)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        # bias terms
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) # hidden_size
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) # hidden_dim*(x_dim+hidden_dim)
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr=1): # 梯度更新
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        # reset diffs to zero
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)


class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct) # state里存有cell_state,以及hidden_state,这与tensorflow中的表示类似
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)

# 这里的一个LSTMNode其是time_step中的一个时间step,作者想要模仿类似于caffe中的输入层与输出层,只不过各层之间是时间先后依赖
class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        # store reference to parameters and to activations
        self.state = lstm_state
        self.param = lstm_param
        # non-recurrent input concatenated with recurrent input
        self.xc = None

    # 底部输入的data(此处所谓的底部,即时间time_step-1)
    def bottom_data_is(self, x_t, s_prev=None, h_prev=None):
        # if this is the first lstm node in the network
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # save data for use in backprop
        self.s_prev = s_prev
        self.h_prev = h_prev

        # concatenate x(t) and h(t-1)
        xc = np.hstack((x_t, h_prev)) # x concatenate, (x_dim+hidden_dim)
        # wg: hidden_dim*(x_dim+hidden_dim), xc: x_dim+hidden_dim, bg: hidden_dim
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg) # g:hidden_dim*1,注意:输入的激活函数为tanh
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi) # i:hidden_dim*1
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf) # f:hidden_dim*1
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo) # o:hidden_dim*1
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f # input* input_gate+ cell_state(t-1)*forget_gate
        self.state.h = self.state.s * self.state.o # h:hidden*1

        self.xc = xc

    # 来自顶部(time_step+1)回传的梯度
    def top_diff_is(self, top_diff_h, top_diff_s):
        # notice that top_diff_s is carried along the constant error carousel
        ds = self.state.o * top_diff_h + top_diff_s # ds: hidden_dim
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds # df:hidden_dim

        # diffs w.r.t. vector inside sigma / tanh function
        di_input = sigmoid_derivative(self.state.i) * di # di_input:hidden_dim
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg

        # diffs w.r.t. inputs
        self.param.wi_diff += np.outer(di_input, self.xc) # 计算两个向量的外积, di_input:hidden_dim,  xc: x_dim+hidden_dim, wi_diff:hidden_dim*(x_dim+hidden_dim)
        self.param.wf_diff += np.outer(df_input, self.xc) # wf_diff:hidden_dim*(x_dim+hidden_dim)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input

        # compute bottom diff
        # 之所以计算对x的梯度,是因为x可能是下面的层
        dxc = np.zeros_like(self.xc) # xc:x_dim+hidden_dim
        dxc += np.dot(self.param.wi.T, di_input) # dxc: (hidden_dim*(x_dim+hidden_dim))^T*hidden_dim = (x_dim+hidden_dim)*1
        dxc += np.dot(self.param.wf.T, df_input) # dxc:  (x_dim+hidden_dim)*1
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # save bottom diffs
        self.state.bottom_diff_s = ds * self.state.f # ds: hidden_dim, f:hidden_dim
        self.state.bottom_diff_h = dxc[self.param.x_dim:] # bottom_diff_h: hidden_dim*1


class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence
        self.x_list = []

    def y_list_is(self, y_list, loss_layer):
        """
        Updates diffs by setting target sequence
        with corresponding loss layer.
        Will *NOT* update parameters.  To update parameters,
        call self.lstm_param.apply_diff()
        """
        assert len(y_list) == len(self.x_list)
        time_step = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_node_list[time_step].state.h, y_list[time_step]) # 计算回归损失
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[time_step].state.h, y_list[time_step]) # 计算最后一层梯度
        # here s is not affecting loss due to h(t+1), hence we set equal to zero
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[time_step].top_diff_is(diff_h, diff_s)
        time_step -= 1

        ### ... following nodes also get diffs from next nodes, hence we add diffs to diff_h
        ### we also propagate error along constant error carousel using diff_s
        while time_step >= 0:
            # 注意: 总loss是所有时间步的loss加起来的损失
            loss += loss_layer.loss(self.lstm_node_list[time_step].state.h, y_list[time_step]) # 计算time_step处的损失
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[time_step].state.h, y_list[time_step])
            diff_h += self.lstm_node_list[time_step + 1].state.bottom_diff_h # 将time+1时间的diff_h累加
            diff_s = self.lstm_node_list[time_step + 1].state.bottom_diff_s
            self.lstm_node_list[time_step].top_diff_is(diff_h, diff_s)
            time_step -= 1

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x) # x:50*1的向量
        if len(self.x_list) > len(self.lstm_node_list):
            # x序列有多长,就用多长的lstmState,但参数永远是共享一套lstm_param
            # need to add new lstm node, create new state mem
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        # get index of most recent x input
        time_step = len(self.x_list) - 1 # idx: time_step
        if time_step == 0:
            # no recurrent inputs yet
            s_prev = None
            h_prev = None
        else:
            s_prev = self.lstm_node_list[time_step - 1].state.s
            h_prev = self.lstm_node_list[time_step - 1].state.h
        # 每层的输入数据为x
        self.lstm_node_list[time_step].bottom_data_is(x, s_prev, h_prev) # x: x_dim=50


