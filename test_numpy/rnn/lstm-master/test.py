
import numpy as np

from lstm import LstmParam, LstmNetwork


class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label): # 平方损失
        return (pred[0] - label) ** 2 #注意:此处pred为hidden隐向量,共有hidden_dim维,但此处只用了1维来预测

    @classmethod
    def bottom_diff(self, pred, label): # 平方损失梯度
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count
    mem_cell_ct = 100
    x_dim = 50
    lstm_param = LstmParam(mem_cell_ct, x_dim)
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5, 0.2, 0.1, -0.5] #这是1个样本的label,分别为每个输入word embedding的打分
    # 为每个y,生成一个x输入向量,x样本序列的长度为4,embedding维度为50
    # 类比例子,可以看成对序列中的每个向量进行pos/neg打分
    input_val_arr = [np.random.random(x_dim) for _ in y_list] # list of ndarray(dim=50), list长度为4
    print("input_val_arr:", input_val_arr)

    for cur_iter in range(210):
        print("iter", "%2s" % str(cur_iter), end=": ")
        for ind in range(len(y_list)):
            # 将每个样本的x放到网络中
            lstm_net.x_list_add(input_val_arr[ind])

        print("y_pred = [" +
              ", ".join(["% 2.5f" % lstm_net.lstm_node_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")
        # 将label放到网络中
        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print("loss:", "%.3e" % loss)
        lstm_param.apply_diff(lr=0.1) # 计算网络的梯度
        lstm_net.x_list_clear() # 每次迭代时,需要将x清空


if __name__ == "__main__":
    example_0()


