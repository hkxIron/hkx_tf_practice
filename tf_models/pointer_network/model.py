import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.framework import tensor_util
from tensorflow.contrib import rnn
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes
# https://github.com/princewen/tensorflow_practice/blob/master/RL/myPtrNetwork/README.md
# https://www.jianshu.com/p/2ad389e91467

LSTMCell = rnn.LSTMCell
MultiRNNCell = rnn.MultiRNNCell

# 返回一个tensor, 如Tensor("my_init_state_0_tiled:0", shape=(3, 2), dtype=float32)
def trainable_initial_state(batch_size,
                            state_size,
                            initializer=None,
                            name="initial_state"):
    flat_state_size = nest.flatten(state_size)

    if not initializer: # None
        flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
    else:
        flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

    names = ["{}_{}".format(name, i) for i in range(len(flat_state_size))]
    tiled_states = []

    for name, size, init in zip(names, flat_state_size, flat_initializer):
        shape_with_batch_dim = [1, size]
        initial_state_variable = tf.get_variable(
            name,
            shape=shape_with_batch_dim,
            initializer=init()
        )

        tiled_state = tf.tile(initial_state_variable,
                              [batch_size, 1], name=(name + "_tiled"))
        tiled_states.append(tiled_state)

    return nest.pack_sequence_as(structure=state_size,
                                 flat_sequence=tiled_states)


"""
sess.run(index_matrix_to_pairs(tf.convert_to_tensor([[-1,2,3],
                                                     [3, 5,6],
                                                     [8, 4,2]])))
                                                     
array([[[ 0, -1], # 0代表第0个样本
        [ 0,  2],
        [ 0,  3]],

       [[ 1,  3],# 1代表第1个样本
        [ 1,  5],
        [ 1,  6]],

       [[ 2,  8],
        [ 2,  4],
        [ 2,  2]]])
"""
def index_matrix_to_pairs(index_matrix):
    # input:
    # [[3,1,2],
    #  [2,3,1]] -> [[[0, 3], [0, 1], [0, 2]], # 0代表第0个样本
    #               [[1, 2], [1, 3], [1, 1]]] # 1代表第1个样本
    #
    # input:
    # [3,1,4] => [[0,3], # [sample_index, seq_index]
    #             [1,1],
    #             [2,4]]
    replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
    rank = len(index_matrix.get_shape())
    if rank == 2:
        # replicated_first_indices:array([[0, 0, 0],
        #                                 [1, 1, 1]])
        replicated_first_indices = tf.tile( # 复制元素
            tf.expand_dims(replicated_first_indices, axis=1),
            multiples=[1, tf.shape(index_matrix)[1]])
    return tf.stack([replicated_first_indices, index_matrix], axis=rank)

class Model(object):
    def __init__(self, config):

        self.task = config.task
        self.debug = config.debug
        self.config = config

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.attention_dim = config.attention_dim
        self.num_layers = config.num_layers

        self.batch_size = config.batch_size

        self.max_enc_length = config.max_enc_length
        self.max_dec_length = config.max_dec_length
        self.num_glimpse = config.num_glimpse
        self.use_terminal_symbol = config.use_terminal_symbol

        self.init_min_val = config.init_min_val
        self.init_max_val = config.init_max_val
        self.initializer = tf.random_uniform_initializer(self.init_min_val, self.init_max_val) # 均匀分布

        self.lr_start = config.lr_start
        self.lr_decay_step = config.lr_decay_step
        self.lr_decay_rate = config.lr_decay_rate
        self.max_grad_norm = config.max_grad_norm

        self.debug_info = {}

        ##############
        # inputs
        ##############

        self.is_training = tf.placeholder_with_default(
            tf.constant(False, dtype=tf.bool),
            shape=(), name='is_training'
        )

        self._build_model()

    def _build_model(self):

        # -----------------定义输入------------------
        # enc_seq:[batch, max_enc_length, input_dim=2], 2代表x,y
        self.enc_seq = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.max_enc_length,2], name='enc_seq')
        # target_seq_index:[batch, max_dec_length]
        self.target_seq_index = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_dec_length], name='target_seq')
        # enc_seq_length:[batch]
        self.enc_seq_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='enc_seq_length')
        # target_seq_length:[batch]
        self.target_seq_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='target_seq_length')

        # ----------------输入处理-------------------
        """
        我们要对输入进行处理，将输入转换为embedding，embedding的长度和lstm的隐藏神经元个数相同。
        这里指的注意的就是 tf.nn.conv1d函数了，这个函数首先会对输入进行一个扩展，然后再调用tf.nn.conv2d进行二维卷积。
        关于该函数的过程可以看代码中的注释或者看该函数的源代码。

        input_dim 是 2，hidden_dim 是 lstm的隐藏层的数量

        # 将输入转换成embedding,一下是根据源码的转换过程：
        # enc_seq :[batch_size,seq_length,input_dim=2] -> [batch_size,1,seq_length,input_dim=2]，在第一维进行维数扩展, 2是x,y坐标, 看成NHWC
        # input_embed : [1,2,256] -> [1,1,2,256] # 在第0维进行维数扩展, 作为filters=[height=1,width=1, in_channel=2, out_channel=256]
        # 所以卷积后的输出为: [batch, 1, seq_length, out_channel]

        # tf.nn.conv1d首先将input和filter进行填充，然后进行二维卷积，因此卷积之后维度为batch * 1 * seq_length * 256
        # 最后还有一步squeeze的操作，从tensor中删除所有大小是1的维度，所以最后的维数为batch * seq_length * 256
        # 即将输入数据:[batch, seq_length, input_dim=2] -> 高维[batch, seq_length, hidden_dim=256], 其实就相当于最后一个维度全连接而己
        # 最后还有一步squeeze的操作，从tensor中删除所有大小是1的维度，所以最后的维数为batch * seq_length * 256
        # embeded_enc_inputs: [batch, seq_length, hidden_dim=256]

        问题是要实现这种变换,为何不用全连接呢?我没想明白
        全连接参数:input_dim*hidden_dim 
        卷积参数:1*1*in_channel*out_channel= input_dim*hidden_dim
        """
        # 将输入转换成embed
        # input_dim 是 2，hidden_dim 是 lstm的隐藏层的数量
        # input_embed: [filter_width=1, input_channel=input_dim=2, output_channel=hidden_dim]
        input_embed = tf.get_variable("input_embed",
            shape=[1, self.input_dim, self.hidden_dim],
            initializer=self.initializer)

        # conv1d是先将tensor升维后经过conv2d处理, 然后再降维
        # enc_seq: [batch, max_enc_length, input_channel=2]
        #       => [batch, in_height=1, in_width=seq_length, in_channels=input_dim=2]
        # input_embed: [filter_width=1, input_channel=input_dim=2, output_channel=hidden_dim]
        #           => [filter_height=1, filter_width=1, in_channels=input_dim=2, out_channels=hidden_dim]
        # conv2d的维度大小: out_size = (img_size+2*pad-filter_size)//stride+1 = (img_size-1)//1+1=img_size,即保持原来大小
        # embeded_enc_inputs: [batch, out_height=1, out_width=seq_length=max_enc_length, output_channel=hidden_dim=256]
        #                  => [batch, seq_length=max_enc_length, hidden_dim=256]
        #
        # 理解:可以看出来,作者选用的是1*1的卷积,即基于单像素在不同通道上的卷积,两个通道分别代表[x,y]坐标
        self.embeded_enc_inputs = tf.nn.conv1d(value=self.enc_seq,
                                               filters=input_embed,
                                               stride=1,
                                               padding="VALID")

        # -----------------encoder------------------
        """
        encoder seq: "1 2 3 4"
        """
        tf.logging.info("Create a model..")
        with tf.variable_scope("encoder"):
            # 构建一个多层的LSTM
            self.enc_cell = LSTMCell(self.hidden_dim, # hidden_dim:256
                initializer=self.initializer)

            if self.num_layers > 1:
                cells = [self.enc_cell] * self.num_layers # num_layers=1,代表有多少层lstm layer
                self.enc_cell = MultiRNNCell(cells)
            # 建立可训练的lstm初始状态
            # 返回一个tensor, 如Tensor("my_init_state_0_tiled:0", shape=(3, 2), dtype=float32), [batch, hidden_dim]
            self.enc_init_state = trainable_initial_state(self.batch_size,
                                                          self.enc_cell.state_size) # hidden_size

            # embeded_enc_inputs: [batch, seq_length=max_enc_length, hidden_dim=256],这里的seq_length其实已经padding到max_sequence长度
            # self.enc_seq_length:[batch]
            # self.encoder_outputs = output_all_hidden_states: [batch_size, seq_length=max_enc_length, hidden_dim]
            # self.enc_final_states = last_cell_and_hidden_state: {c: [batch_size, hidden_size], h:[batch_size, hidden_size]}
            self.encoder_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
                self.enc_cell, # lstm cell
                self.embeded_enc_inputs, #
                self.enc_seq_length, # [batch]
                self.enc_init_state) # [batch, hidden_dim]

            # 给最开头添加一个结束标记，同时这个标记也将作为decoder的初始输入
            # first_decoder_input:[batch_size, seq_length=1, hidden_dim], 代表:EOS 或 SOS
            self.first_decoder_input = tf.expand_dims( # 其实它的值可以不为0,而是随机一个
                #input=trainable_initial_state(self.batch_size, self.hidden_dim, initializer=tf.initializers.truncated_normal(), name="first_decoder_input"), # 值为全0的tensor,Tensor("my_init_state_0_tiled:0", shape=(3, 2), dtype=float32)
                input=trainable_initial_state(self.batch_size, self.hidden_dim, name="first_decoder_input"), # 值为全0的tensor,Tensor("my_init_state_0_tiled:0", shape=(3, 2), dtype=float32)
                axis=1
            )

            # 0 index indicates terminal, 第0个元素代表 SOS(也可以称为EOS)
            # 比如原来序列为:"1 2 3 4", 现在encoder_outputs:"SOS 1 2 3 4"
            # first_decoder_input: [batch_size, seq_length=1, hidden_dim]
            # encoder_outputs: [batch_size, seq_length=max_enc_length, hidden_dim]
            #               => [batch_size, 1+seq_length, hidden_dim]
            self.encoder_outputs = tf.concat(values=[self.first_decoder_input, self.encoder_outputs], # 在encoder_outputs最前面插入 EOS, 即encoder seq:"EOS 1 2 3 4"
                                             axis=1)

        # -----------------decoder 训练--------------------
        """
        在seq2seq中: 知识就是力量 
                 => <SOS> knowledge is power.
        即 SOS -> knowledge
           knowledge -> is          
           is -> power
           power -> EOS.
        seq2seq中decoder的输入是另外一种decoder_embedding,即一般不与encoder共享encoder_embedding
        
        与seq2seq不同的是，pointer-network的decoder的输入并不是target序列的单独的embedding，
        而是根据target序列的值选择相应位置的encoder的hidden输出作为decoder的输入,而不是与seq2seq中一样embedding_lookup,
        同时预测的目标也是输入中序列的元素下标,
        所以在pointer-network中decoder阶段需要对encoder-output进行stop_gradient。
        
        1 2 3 4 =>
                  <SOS> 1 4 2 1
                  
        即 SOS     => 1
           [x1,y1] => 4
           [x4,y4] => 2
           [x2,y2] => 1
           [x1,y1] => EOS
        注意:EOS并非输入的数据,而是强制加入预测阶段,以使输出序列终止
        
        我们知道encoder的输出长度在添加了开始SOS(即0)输出之后形状为:[batch, 1+max_enc_seq_length, hidden_dim]。
        现在假设我们拿第一条记录进行训练,第一条记录encoder输入的是点[1,2,3,4]的xy坐标序列:
        [[x1,y1],
         [x2,y2],
         [x3,y3],
         [x4,y4],
        ]，
        相应的预测序列是:[1,4,2]，那么decoder依次的输入[SOS,1,4,2,1]:
        self.enc_outputs[0][0], 0代表 "SOS"
        self.enc_outputs[0][1], 注意不是embedding_lookup
        self.enc_outputs[0][4], 
        self.enc_outputs[0][2], 
        self.enc_outputs[0][1], 
        
        那么如何根据target序列来选择encoder的输出呢，这里就要用到我们刚刚定义的
        index_matrix_to_pairs函数以及gather_nd函数：
        """
        with tf.variable_scope("decoder"):
            # target_seq:
            # [[3,1,2], 第0个样本的目标预测序列
            #  [2,3,1]] 第0个样本的目标预测序列
            # ->
            # target_idx_pairs: [batch, max_dec_length, indexs=2], 其中indexs=[sample_index_in_batch, seq_index]
            # [[[0, 3], [0, 1], [0, 2]], 含义:batch中共有2个样本,第0个样本序列长度为3,第0个样本的输入序列为:[3,1,2]
            #  [[1, 2], [1, 3], [1, 1]]]

            # 将target_index转化为 (batch_index, target_index) 对
            # target_seq_index: [batch=20, max_dec_length=10]
            # target_idx_pairs: [batch=20, max_dec_length=10, indexs=2], 其中indexs=[sample_index_in_batch, seq_index]
            self.target_idx_pairs = index_matrix_to_pairs(self.target_seq_index)

            # encoder_outputs:    [batch, 1+max_enc_sequence, hidden_dim], 第0个元素是 EOS
            # target_idx_pairs:   [batch, max_dec_length, indexs=2], 其中indexs=[sample_index_in_batch, seq_index]
            # embeded_dec_inputs: [batch, max_dec_length, hidden_dim], 注意:这里seq_length=max_dec_length,而不是1+max_enc_seq, 因为根据pointer-network,并非所有输入结点都会在输出序列中
            #
            # gather_nd完成的功能就是从enc_outputs中选出target_seq中对应的index的向量,组成embed_dec_inputs
            self.embeded_dec_inputs = tf.stop_gradient( # 共用encoder_outputs,阻止梯度传回到encoder
                tf.gather_nd(self.encoder_outputs, self.target_idx_pairs))

            """
            在此处,我们可以看到ptr-net与MT中的seq2seq区别,seq2seq中的decoder输入一般是另外的embedding输入,
            而在ptr-net中,直接将target_seq_ids所对应的encoder的hidden输出copy过来当作decoder的inputs!
            然后用这些decoder中的inputs来预测target_seq_ids(softmax选最大值)

            因此,此处不能有梯度回传到encoder
            """
            self.debug_info["target_seq"] = self.target_seq_index
            self.debug_info["enc_outputs"] = self.encoder_outputs
            self.debug_info["target_idx_pairs"] = self.target_idx_pairs
            self.debug_info["embeded_dec_inputs"] = self.embeded_dec_inputs

            """
            由于decoder的输出变成了原先的target序列的长度+1，
            因此我们要在每个target后面补充一个结束标记，我们补充1作为结束标记：
            """
            # 给target最后一维增加结束标记,数据都是从1开始的，所以结束也是回到1
            # tiled_zero_idxs:[batch, 1],注意,补1的地方的值均为0
            tiled_zero_idxs = tf.tile(tf.zeros([1, 1], dtype=tf.int32),  # 加一个O的index代表EOS,即需要你预测 encoder_outputs中添加到最前面的 SOS终止符
                                      multiples=[self.batch_size, 1],
                                      name="tiled_zero_idxs")
            # target_seq_index:[batch, max_dec_length]
            # add_terminal_target_index_seq: [batch, max_dec_length+1], 注意:target id中结束标记加在末尾,加的index=0,注意此处必须是index=0,不能是其它的index
            self.add_terminal_target_index_seq = tf.concat([self.target_seq_index, tiled_zero_idxs],
                                                           axis=1) # 即 "1 4 2 1" -> "1 4 2 1 EOS"

            #如果使用了结束标记的话，要给encoder的输出拼上开始状态，同时给decoder的输入拼上开始状态
            # embeded_dec_inputs: [batch, max_dec_length, hidden_dim]
            #                  => [batch, 1+max_dec_length, hidden_dim],
            #                  即 decoder inputs: "SOS 1 4 2 1"  decoder outputs:"1 4 2 1 EOS'
            self.embeded_dec_inputs = tf.concat([self.first_decoder_input,
                                                 self.embeded_dec_inputs],
                                                axis=1) # embedding中结束标记加在首位
            """
            从以上可以看出, target_id中结束标记加在末尾,而embedding中结束标记加在首位,相当于embedding(i) => target(i+1)
            """
            # 建立一个多层的lstm网络
            self.dec_cell = LSTMCell(self.hidden_dim, initializer=self.initializer)

            if self.num_layers > 1:
                cells = [self.dec_cell] * self.num_layers
                self.dec_cell = MultiRNNCell(cells)

            # encoder的最后的状态作为decoder的初始状态
            # self.enc_final_states = last_cell_and_hidden_state: {c: [batch_size, hidden_size], h:[batch_size, hidden_size]}
            dec_state = self.enc_final_states

            # 预测的序列
            self.predict_indexes = []
            # 预测的softmax序列，用于计算损失
            self.predict_indexes_distribution = []

            """
            对于decoder来说，这里我们每次每个batch只输入一个值，然后使用循环来实现整个decoder的过程：
            即对于每条样本的每个输出时刻i,都需要计算与每个输入时刻j的attention,因此时间复杂度为o(m*n),运行速度较慢
            "SOS 1 4 2 1" => "1 4 2 1 EOS"
            """
            # 训练self.max_dec_length  + 1轮，每一轮输入batch * hidden_dim
            for i in range(self.max_dec_length + 1):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                # embeded_dec_inputs: [batch, 1+max_dec_length, hidden_dim], 即 "SOS 1 4 2 1"
                # cell_input:[batch, hidden_dim]
                cell_input = tf.squeeze(self.embeded_dec_inputs[:, i, :])  # [batch, hidden_dim]
                # output_i:[batch, hidden_dim], dec_state: {c: [batch_size, hidden_size], h: [batch_size, hidden_size]}
                output_i, dec_state = self.dec_cell(inputs=cell_input, state=dec_state)  # lstm经过一个时间步后的output: [batch, hidden],由于并没有多个timestep,所以不需要dynamic_rnn

                # 使用pointer机制选择得到softmax的输出，idx_softmax_probility:[batch, max_enc_length + 1]
                # 论文中decoder时刻i对所有输入时刻j的attention系数: u(i,j) = V^T*tanh(W1*ej+W2*di), j in(1,...,n)
                # encoder_outputs:[batch_size, 1+encoder_seq_length, hidden_dim], encoder时刻中的所有j, encoder seq:"EOS 1 2 3 4"
                # output_i:[batch, hidden_dim], decoder时刻中的i
                # idx_softmax_probility:[batch, 1+encoder_seq_length], 当前decoder时刻中的i对encoder所有时刻j的attention系数
                idx_softmax_probility = self.choose_index_probility(self.encoder_outputs, output_i)
                # 选择每个batch中最大的id, [batch]
                # idx_softmax_probility:[batch, 1+encoder_seq_length]
                # idx:[batch]
                idx = tf.argmax(idx_softmax_probility, axis=1, output_type=dtypes.int32)

                # decoder的每个输出的softmax序列
                # idx_softmax_probility:[batch, 1+max_enc_length]
                # predict_indexes_distribution: [max_dec_length+1, batch, 1+max_enc_length]
                self.predict_indexes_distribution.append(idx_softmax_probility)

                # decoder的每个输出的id
                # idx:[batch]
                # predict_indexes:[max_dec_length+1, batch]
                self.predict_indexes.append(idx)

            # "1 4 2 1 EOS"
            self.predict_indexes = tf.convert_to_tensor(self.predict_indexes) # list-> tensor, [max_dec_length+1, batch]
            # predict_indexes_distribution中的下标: "1 4 2 1 EOS"
            self.predict_indexes_distribution = tf.convert_to_tensor(self.predict_indexes_distribution) # list->tensor, [max_dec_length+1, batch, 1+max_enc_length]

        # ----------------loss------------------
        with tf.variable_scope("loss"):
            # # 我们计算交叉熵来作为我们的损失
            # # -sum(y * log y')
            # # 首先我们要对我们的输出进行一定的处理，首先我们的target的维度是batch * self.max_dec_length * 1，
            # # 而训练或预测得到的softmax序列是 (self.max_dec_length +1)* batch * (self.max_enc_length + 1)
            # # 所以我们先去掉预测序列的最后一行，然后进行transpose，再转成一行
            # # 对实际的序列，我们先将其转换成one-hot，再转成一行，随后便可以计算损失
            #
            # self.dec_pred_logits = tf.reshape(
            #     tf.transpose(tf.squeeze(self.predict_indexes_distribution), [1, 0, 2]), [-1])  # B * D * E + 1
            # self.dec_inference_logits = tf.reshape(
            #     tf.transpose(tf.squeeze(self.infer_predict_indexes_distribution), [1, 0, 2]),
            #     [-1])  # B * D * E + 1
            # self.dec_target_labels = tf.reshape(tf.one_hot(self.add_terminal_target_index_seq, depth=self.max_enc_length+ 1), [-1])
            #
            # self.loss = -tf.reduce_sum(self.dec_target_labels * tf.log(self.dec_pred_logits))
            # self.inference_loss = -tf.reduce_mean(self.dec_target_labels * tf.log(self.dec_inference_logits))

            # predict_indexes_distribution中的下标: "1 4 2 1 EOS"
            # predict_indexes_distribution: [max_dec_length+1, batch, max_enc_length + 1]
            # pred_indexes_distribution: [max_dec_length+1, batch, max_enc_length + 1]  or [max_dec_length, batch, max_enc_length + 1
            if self.use_terminal_symbol: # 不知为何使用了use_terminal_symbol后,训练时loss无法下降
                pred_indexes_distribution = self.predict_indexes_distribution
            else:
                pred_indexes_distribution = self.predict_indexes_distribution[:-1] # [:-1], 计算loss时去掉最后的结束符 EOS

            # training_logits: [max_dec_length+1 or max_dec_length, batch, max_enc_length + 1], 未去掉最后的EOS然后计算loss
            #               => [batch, max_dec_length+1 or max_dec_length, max_enc_length + 1]
            training_logits = tf.identity(tf.transpose(pred_indexes_distribution, perm=[1,0,2])) # [:-1], 计算loss时去掉最后的结束符 EOS

            # target_seq_index:[batch, max_dec_length], 下标:"1 4 2 1"
            if self.use_terminal_symbol:
                # target_seq_indexs:[batch, max_dec_length+1]
                target_seq_indexs = tf.identity(self.add_terminal_target_index_seq) # 有终止符, 下标: "1 4 2 1 EOS"
                # target_seq_length:[batch], 记录了每样本需要预测的真正长度
                # max_dec_length:[1], scalar
                # masks: [batch, max_dec_length+1], 用sequence_mask补零
                masks = tf.sequence_mask(self.target_seq_length+1, maxlen=self.max_dec_length+1, dtype=tf.float32, name="masks")
            else:
                # target_seq_indexs:[batch, max_dec_length]
                target_seq_indexs = tf.identity(self.target_seq_index)
                # target_seq_length:[batch], 记录了每样本需要预测的真正长度
                # max_dec_length:[1], scalar
                # masks: [batch, max_dec_length], 用sequence_mask补零
                masks = tf.sequence_mask(self.target_seq_length, maxlen=self.max_dec_length, dtype=tf.float32, name="masks")

            """
            一条样本的 target_seq_indexs 如下:
            1 4 2 1 
            即已经包含 +1了
            即 SOS     => 1
               [x1,y1] => 4
               [x4,y4] => 2
               [x2,y2] => 1
               [x1,y1] => EOS 
            
            注意:training_logits已经去掉了最后的结束符
            """
            # training_logits: [batch, max_dec_length or max_dec_length+1, max_enc_length + 1]
            # loss:[1],此处的loss为所有序列拼起来的平均值, 防止序列越长,loss越大,从而使模型倾向于选择更短的序列
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=training_logits, # [batch_size, max_dec_length or max_dec_length+1, max_enc_length + 1], 1代表EOS,其它的代表 1 2 3 4
                targets=target_seq_indexs, # [batch_size, max_dec_length or max_dec_length+1]
                weights=masks # [batch_size, max_dec_length or max_dec_length+1]
            )
            self.optimizer = tf.train.AdamOptimizer(self.lr_start)
            self.train_op = self.optimizer.minimize(self.loss)


        # ----------------------decoder inference----------------------
        # 预测输出的id序列
        self.infer_predict_indexes = []
        # 预测输出的softmax序列
        self.infer_predict_indexes_distribution = []
        with tf.variable_scope("decoder", reuse=True):
            # self.enc_final_states = last_cell_and_hidden_state: {c: [batch_size, hidden_size], h:[batch_size, hidden_size]}
            dec_state = self.enc_final_states
            # 预测阶段最开始的输入是之前定义的初始输入
            # perdict_decoder_input:[batch_size, seq_length=1, hidden_dim]
            self.predict_decoder_input = self.first_decoder_input  # SOS
            # 注意:此处与train阶段的decoder的不同
            for i in range(self.max_dec_length + 1):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                """
                self.embeded_dec_inputs = tf.concat([self.first_decoder_input, self.embeded_dec_inputs], axis=1) # embedding中结束标记加在首位
                注意:此处predict_decoder_input与embed_dec_inputs不同,此处并没有target_id组成的序列的输入
                """
                # predict_decoder_input:[batch_size, seq_length=1, hidden_dim]
                #                     =>[batch_size, hidden_dim]
                self.predict_decoder_input = tf.squeeze(self.predict_decoder_input)  # [batch, 1, hidden] -> [batch, hidden]

                # 因为这里是按时间time展开的,所以output里没有timestep
                # output_i:[batch_size, hidden_dim]
                # dec_state = {c: [batch_size, hidden_size], h:[batch_size, hidden_size]}
                output_i, dec_state = self.dec_cell(inputs=self.predict_decoder_input, state=dec_state)  # output:[batch, hidden]

                # 同样根据pointer机制得到softmax输出
                # encoder_outputs: [batch_size, 1+max_enc_length, hidden_dim]
                # output_i:[batch_size, hidden_dim]
                # idx_softmax_probility: [batch, 1+max_enc_length]
                idx_softmax_probility = self.choose_index_probility(self.encoder_outputs, output_i)

                # 选择最大的那个id
                idx = tf.argmax(idx_softmax_probility, axis=1, output_type=dtypes.int32)  # [batch]

                # 将选择的id转换为pair
                # idx:[batch]
                # idx_pairs:[batch, 2], 第一列为sample_index, 第二列为seq_index
                idx_pairs = index_matrix_to_pairs(idx)

                # 更新predict_decoder_input: 选择下一个时刻的输入,此处亦与train decoder阶段不同,在train decoder中并不需要更新
                # encoder_outputs:[batch, 1+seq_length, hidden_dim]
                # idx_pairs:[batch, 2], 第一列为sample_index, 第二列为seq_index
                # predict_decoder_input:[batch, 1, hidden]
                self.predict_decoder_input = tf.stop_gradient(tf.gather_nd(self.encoder_outputs, idx_pairs)) # 更新下一个decoder的input,此处需要阻断梯度,不需要将梯度传回encoder

                # decoder的每个输出的id
                # idx:[batch]
                # infer_predict_indexes:[max_dec_length+1, batch]
                self.infer_predict_indexes.append(idx)

                # decoder的每个输出的softmax序列
                # idx_softmax_probility: [batch, 1+enc_max_length]
                # infer_predict_indexes_distribution:[max_dec_length+1, batch, 1+max_enc_length]
                self.infer_predict_indexes_distribution.append(idx_softmax_probility)

            # infer_predict_indexes:[max_dec_length+1, batch]
            self.infer_predict_indexes = tf.convert_to_tensor(self.infer_predict_indexes, dtype=tf.int32)
            # infer_predict_indexes_distribution:[max_dec_length+1, batch, 1+max_enc_length]
            self.infer_predict_indexes_distribution = tf.convert_to_tensor(self.infer_predict_indexes_distribution, dtype=tf.float32)

    def train(self, sess, batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.enc_seq: batch['enc_seq'],
                     self.enc_seq_length: batch['enc_seq_length'],
                     self.target_seq_index: batch['target_seq'],
                     self.target_seq_length: batch['target_seq_length']}
        debug_info = ""
        if self.config.debug:
            _, loss, debug_info = sess.run([self.train_op, self.loss, self.debug_info], feed_dict=feed_dict)
        else:
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss, debug_info

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.enc_seq: batch['enc_seq'],
                     self.enc_seq_length: batch['enc_seq_length'],
                     self.target_seq_index: batch['target_seq'],
                     self.target_seq_length: batch['target_seq_length']}
        loss= sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def infer(self, sess, batch):
        # infer与eval相同,但只要predict,而不需要loss
        feed_dict = {self.enc_seq: batch['enc_seq'],
                     self.enc_seq_length: batch['enc_seq_length'],
                     self.target_seq_index: batch['target_seq'],
                     self.target_seq_length: batch['target_seq_length']}
        predict = sess.run([self.infer_predict_indexes], feed_dict=feed_dict)
        return predict


    def attention(self, ref_encoders, query, with_softmax, scope="attention"):
        """
        从后来transformer的角度来看,Key=ref, Value=ref, Query=query

        u_i = v^T*tanh(Wq*q + W_ref*ri ), (if i!=pai(j) for all j<i)
            = V^T*tanh(W_decoder*query + W_encoder* encoder_i) + bias
        attention_i = softmax(u_i)

        :param ref_encoders: encoder的输出, [batch, max_enc_length, hidden_dim]
        :param query: decoder的输入, [batch, hidden]
        :param with_softmax:
        :param scope:
        :return: [batch, max_enc_length]
        """
        with tf.variable_scope(scope):
            W_encoder = tf.get_variable("W_e", [self.hidden_dim, self.attention_dim], initializer=self.initializer) # [hidden, atten_dim]
            W_decoder = tf.get_variable("W_d", [self.hidden_dim, self.attention_dim], initializer=self.initializer) # [hidden, atten_dim]

            # query: [batch, hidden]
            # W_decoder:[hidden, attention_dim]
            # decoder_portion: [batch, attention_dim=20]
            decoder_portion = tf.matmul(query, W_decoder) # 将decoder里的query转换到attention空间

            scores = [] # [max_enc_length+1, batch]
            # v_blend:[atten_dim,1]
            v_blend = tf.get_variable("v_blend", [self.attention_dim, 1], initializer=self.initializer)
            # bais_blend:1
            bais_blend = tf.get_variable("bais_v_blend", [1], initializer=self.initializer)
            # 对于输入的每个时刻T_i
            for i in range(self.max_enc_length + 1):
                # ref:[batch, max_enc_length, hidden]
                #    => squeeze: [batch, hidden]
                # W_encoder:[hidden, atten_dim]
                # refi: [batch, atten_dim],我感觉这里存在这很多重复计算
                refi = tf.matmul(tf.squeeze(ref_encoders[:, i, :]), W_encoder) # 此行是将ref_encoder转换到attention空间

                # decoder_portion: [batch, atten_dim=20]
                # refi: [batch, atten_dim]
                # v_blend:[atten_dim,1]
                # ui:[batch, 1]
                ui = tf.matmul(tf.nn.tanh(decoder_portion + refi), v_blend) + bais_blend # 公式: V^T*tanh(W_decoder*query + W_encoder* encoder_hidden_i) + bias

                # scores:[max_enc_length+1, batch]
                scores.append(tf.squeeze(ui))

            # scores: [max_enc_length+1, batch]
            #      => [batch, max_enc_length+1]
            scores = tf.transpose(scores, perm=[1,0]) # scores的含义,当前decoder时间步i对于所有encoder_j的注意力分数
            if with_softmax:
                return tf.nn.softmax(scores, axis=1) # [batch, max_enc_length+1]
            else:
                return scores # [batch, max_enc_length+1]


    """ 
        在论文中还提到一个词叫做glimpse function，他首先将上面式子中的q进行了处理，公式如下：
        
        
        计算经过与输入对齐之后的query
        
        ref: [batch, max_enc_encoder, hidden]
        query:[batch, hidden]
        
        :return [batch, hidden], 用query对ref的hidden隐向量进行attention加权
    """
    def glimpse_fn(self, ref, query, scope="glimpse"):
        # query:[batch, hidden]
        # ref: [batch, max_enc_encoder, hidden]
        # p:[batch, max_enc_encoder], 每行均为一个概率分布
        p = self.attention(ref, query, with_softmax=True, scope=scope)
        # p_alignments: [batch, max_enc_encoder, 1]
        p_alignments = tf.expand_dims(p, axis=2)
        # p_alignments: [batch, max_enc_encoder, 1]
        # ref: [batch, max_enc_encoder, hidden]
        # return [batch, hidden], 用query对ref的各时间步的hidden隐向量进行attention加权
        return tf.reduce_sum(p_alignments * ref, axis=[1], keep_dims=False)


    """ 
    ref: [batch, max_enc_encoder, hidden]
    query:[batch, hidden]
    """
    def choose_index_probility(self, ref, query):
        if self.num_glimpse > 0: # 对齐query
            query = self.glimpse_fn(ref, query)
        return self.attention(ref, query, with_softmax=True, scope="attention")

if __name__ == "__main__":
    index=tf.constant([1,4,5])
    pairs = index_matrix_to_pairs(index)
    with tf.Session() as sess:
        print("paris:", sess.run(pairs))

    batch_size = 3
    state_size = 2
    enc_init_state = trainable_initial_state(batch_size, state_size, tf.zeros_initializer, "my_init_state")
    print(enc_init_state) # Tensor("my_init_state_0_tiled:0", shape=(3, 2), dtype=float32)
