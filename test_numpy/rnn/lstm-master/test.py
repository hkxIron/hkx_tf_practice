import numpy as np

from lstm import LstmParam, LstmNetwork

class ToyLossLayer:
    """
    Computes square loss with first element of hidden layer array.
    """
    @classmethod
    def loss(self, pred, label): #loss
        return (pred[0] - label) ** 2

    @classmethod
    def bottom_diff(self, pred, label):#gradient
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

def example_0():
    # learns to repeat simple sequence from random inputs
    np.random.seed(0)

    # parameters for input data dimension and lstm cell count 
    mem_cell_ct = 100  #100个lstm节点
    x_dim = 50  #输入值的维度
    concat_len = x_dim + mem_cell_ct  #150
    lstm_param = LstmParam(mem_cell_ct, x_dim)  #
    lstm_net = LstmNetwork(lstm_param)
    y_list = [-0.5,0.2,0.1, -0.5] #4
    input_val_arr = [np.random.random(x_dim) for _ in y_list]  #产生4个x_dim维的向量,每个向量维度为:50
    #即为每个向量x训练到y的模型

    for cur_iter in range(100): #训练100次
        print "cur iter: ", cur_iter
        for ind in range(len(y_list)):
            lstm_net.x_list_add(input_val_arr[ind])
            print "y_pred[%d] : %f" % (ind, lstm_net.lstm_node_list[ind].state.h[0])

        loss = lstm_net.y_list_is(y_list, ToyLossLayer)
        print "loss: ", loss
        lstm_param.apply_diff(lr=0.1)
        lstm_net.x_list_clear()

if __name__ == "__main__":
    example_0()

