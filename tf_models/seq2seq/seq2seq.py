"""
注意:此实现中的decoder部分与原seq2seq-attention论文neural machine translation by jointly learning to align and translate 2015
中不太一样,原论文中decoder_hidden_j都是依赖于y_(i-1),C_i,即S_i = g(s_(i-1), y_(i-1), c_i),
而此code中S_i仅依赖于 S_(i-1), 而只是在后面的全连接层才用到C_i与y_(i-1)
所以感觉并非标准的seq2seq-attention
"""

#! -*- coding: utf-8 -*-
# blog: https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247491314&idx=1&sn=3e22d4a6d732b0877fdc567d2bce1076&chksm=96e9c172a19e48646005da05e143751aa9012c141dd1cf9846a2b418cbf854c7d343013105a1&scene=21#wechat_redirect
# origin code: https://github.com/bojone/seq2seq/blob/master/seq2seq.py
import numpy as np
#import pymongo
from tqdm import tqdm
import os,json
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam

min_count = 2
maxlen = 400
batch_size = 64
epochs = 100
char_embedding_size = 128
#db = pymongo.MongoClient().text.news # 我的数据存在mongodb中

# 0: mask (padding)
# 1: unk
# 2: start
# 3: end
MASK = 0
UNK = 1
START = 2
END = 3
SPECIAL_TOKEN_LEN = len([MASK, UNK, START, END]) # mask, unk, start, end
SEP="[SEP]" #正文与标题之间的分隔符

train_data_path = "train_data.txt"

if os.path.exists('seq2seq_config.json'):
    chars,id2char,char2id = json.load(open('seq2seq_config.json'))
    id2char = {int(i):j for i,j in id2char.items()}
else:
    chars = {}
    #for a in tqdm(db.find()):
    for line in open(train_data_path, "r", encoding="utf8"):
        arr = line.strip().split(SEP)
        # 纯文本,未分词
        for w in arr[0]:
            chars[w] = chars.get(w,0) + 1
        for w in arr[1]:
            chars[w] = chars.get(w,0) + 1
    # 频次过滤
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+SPECIAL_TOKEN_LEN:j for i,j in enumerate(chars)}
    char2id = {j:i for i,j in id2char.items()}
    json.dump([chars,id2char,char2id], open('seq2seq_config.json', 'w'))
    print("json dump seq2seq_config file.")

VOCAB_SIZE_WITH_SPECIAL = len(chars) + SPECIAL_TOKEN_LEN
print("char len:", VOCAB_SIZE_WITH_SPECIAL)

def str2id(s, start_end=False):
    # 文字转整数id
    if start_end: # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen-2]]
        ids = [START] + ids + [END]
    else:
    # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return ids


def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(batch): # batch:二维list
    # padding至batch内的最大长度的id序列
    max_length = max([len(sample) for sample in batch])
    return [sample + [MASK] * (max_length - len(sample)) for sample in batch]

def data_generator():
    # 数据生成器
    X,Y = [],[]
    while True:
        # 数据格式:正文[SEP]标题
        for line in open(train_data_path, "r", encoding="utf8"):
            arr = line.strip().split(SEP)
            X.append(str2id(arr[0]))
            Y.append(str2id(arr[1], start_end=True))
            if len(X) == batch_size:
                X = np.array(padding(X))
                Y = np.array(padding(Y))
                yield [X,Y], None
                X,Y = [],[]


# 搭建seq2seq模型
x_in = Input(shape=(None,)) # [batch, max_enc_length]
y_in = Input(shape=(None,)) # [batch, max_dec_length]
x = x_in
y = y_in
# 通过lambda得到layer, x_mask其实也可以通过每个样本length来得到
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, axis=2), MASK), 'float32'))(x) # [batch, max_enc_length, 1]
y_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, axis=2), MASK), 'float32'))(y) # [batch, max_dec_length, 1]

# 选出每个样本里出现的那些词汇
def to_one_hot(x_and_mask:list): # 输出一个词表大小的向量，来标记该词是否在当前文章出现过
    x, x_mask = x_and_mask # x:[batch, max_enc_length] x_mask:[batch, max_dec_length,1]
    x = K.cast(x, 'int32')
    # x_one_hot: [batch, max_enc_length, vocab_size]
    x_one_hot = K.one_hot(x, num_classes=VOCAB_SIZE_WITH_SPECIAL)
    # x_one_hot: [batch, max_enc_length, vocab_size] x_mask:[batch, max_dec_length,1]
    # x_sum:[batch, 1, vocab_size]
    x_sum = K.sum(x_mask * x_one_hot, axis=1, keepdims=True)
    # x_out:[batch, 1, vocab_size]
    x_out = K.cast(K.greater(x_sum, MASK), 'float32')
    #原始: x_out = K.cast(K.greater(x_sum, 0.5), 'float32')
    return x_out


class ScaleShift(Layer):
    """缩放平移变换层（Scale and shift）
       y = s*x+t
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    # build先执行,call是调用时才执行
    def build(self, input_shape):
        # input_shape:(batch, 1, vocab_size)
        # kernel_shape:(1, 1, vocab_size)
        # log_scale:(1,1,vocab_size)
        kernel_shape = (1,)*(len(input_shape)-1) + (input_shape[-1],) # [1,1,1, hidden]
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')
    def call(self, inputs):
        # log_scale:[1,1,vocab_size], inputs:[batch, 1, vocab_size] shift:[batch, 1, vocab_size]
        x_outs = K.exp(self.log_scale) * inputs + self.shift  # exp(log(x)) = x, 这样出来的x全为正数,即 w*x + shift
        return x_outs

# to_one_hot = Lambda(to_one_hot)([x, x_mask])
# to_one_hot: [batch, 1, vocab_size]
# x_one_hot_layer:[batch, 1, vocab_size]
#x_one_hot = to_one_hot([x, x_mask])
"""
to_one_hot为被调用的function, [x, x_mask]为to_one_hot需要输入的数据,
与直接调用to_one_hot([x, x_mask])不同,Lambda会返回Layer object
"""
x_one_hot_layer = Lambda(function=to_one_hot)([x, x_mask])
# x_prior: [batch, 1, vocab_size]
x_prior = ScaleShift()(x_one_hot_layer) # 调用call方法,学习输出的先验分布（标题的字词很可能在文章出现过）, x_prior = s*x +t

# 注意此处:encoder与decoder共享embedding, x -> y
embedding = Embedding(input_dim=VOCAB_SIZE_WITH_SPECIAL, output_dim=char_embedding_size)
# x: [batch, max_enc_length, char_embedding]
x_embedding = embedding(x)
# y: [batch, max_dec_length, char_embedding]
y_embedding = embedding(y)

# encoder，双层双向LSTM
"""
x = Bidirectional(CuDNNLSTM(char_size/2, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(char_size/2, return_sequences=True))(x)
"""
encoder_embedding_size = char_embedding_size//2
# x_bi_lstm1: [batch, max_enc_length, char_embedding]
x_bi_lstm1 = Bidirectional(layer=LSTM(units = encoder_embedding_size, return_sequences=True), merge_mode='concat')(x_embedding) # 先调用 LSTM层,然后调用Bidirectional层
# x_bi_lstm2: [batch, max_enc_length, char_embedding]
x_bi_lstm2 = Bidirectional(layer=LSTM(units=encoder_embedding_size, return_sequences=True), merge_mode='concat')(x_bi_lstm1)

# decoder，双层单向LSTM, 注意是单向
""" 
注意:此实现中的decoder部分与原seq2seq-attention论文neural machine translation by jointly learning to align and translate 2015
中不太一样,原论文中decoder_hidden_j都是依赖于y_(i-1),而此处并没有,所以感觉并非标准的seq2seq-attention

此处S_i = g(S_(i-1))
"""
# y: [batch, max_dec_length, char_embedding]
y_lstm1 = LSTM(units=char_embedding_size, return_sequences=True)(y_embedding) # 此处decoder阶段竟然直接用lstm进行了计算,没有用到C_i,y_(i-1)
y_lstm2 = LSTM(units=char_embedding_size, return_sequences=True)(y_lstm1)


class Interact(Layer):
    """交互层，负责融合encoder和decoder的信息
    """
    def __init__(self, **kwargs):
        super(Interact, self).__init__(**kwargs)

    """
     y: [batch, max_dec_length, char_embedding]
     x: [batch, max_enc_length, char_embedding//2] 
     x_mask: [batch, max_enc_length, 1]
    """
    def build(self, input_shape):
        in_dim = input_shape[0][-1] # y.shape[-1] = char_embedding
        out_dim = input_shape[1][-1] # x.shape[-1] = char_embedding
        # kernel: [char_embedding, char_embedding]
        self.kernel = self.add_weight(name='kernel', #  transformer中的Wq
                                      shape=(in_dim, out_dim),
                                      initializer='glorot_normal')

    """
     y -> query: [batch, max_dec_length, char_embedding]
     x -> value: [batch, max_enc_length, char_embedding] 
     x_mask: [batch, max_enc_length, 1], 0-1矩阵
    """
    def call(self, inputs):
        # query:[batch, max_dec_length, char_embedding],即decoder hidden - S_i
        # value:[batch, max_enc_length, char_embedding],即encoder hidden - h_j
        # value_mask:[batch, max_enc_length, 1], 0-1矩阵
        query, value, value_mask = inputs
        key = value # 对于这种attention而言,key与value相同

        # 对于每个样本计算所有timestep里最大的value
        # value:[batch, max_enc_length, char_embedding]
        # max_pooling_over_time:[batch, 1, 1],
        max_pooling_over_time = K.max(value - (1. - value_mask) * 1e10, axis=1, keepdims=True) # maxpooling1d
        # max_pooling_over_time:[batch, max_dec_length, char_embedding]
        max_pooling_over_time_extend = max_pooling_over_time + K.zeros_like(query[:, :, :]) # 将mv重复至“query的timesteps”份

        # 下面几步只是实现了一个乘性attention
        # qw: [batch, max_dec_length, char_embedding]
        qw = K.dot(query, self.kernel) # q: [batch, max_dec_length, char_embedding], kernel:[char_embedding, char_embedding]
        # qw: [batch, max_dec_length, char_embedding], k: [batch, max_enc_length, char_embedding]
        # 所谓的batch_dot: out = tf.matmul(x1, x2, adjoint_a=None, adjoint_b=True)
        # attention_mat: [batch, max_dec_length, max_enc_length] # 每一个decoder step对所有的encoder step进行attention
        attention_mat = K.batch_dot(qw, key, axes=[2, 2]) / 10. # max_dec_length=10
        attention_mat -= (1. - K.permute_dimensions(value_mask, [0, 2, 1])) * 1e10
        # attention_mat: [batch, max_dec_length, max_enc_length]
        attention_mat = K.softmax(attention_mat, axis= -1)
        # out = tf.matmul(x1, x2, adjoint_a=None, adjoint_b=None)
        # attention_mat: [batch, max_dec_length, max_enc_length]
        # value:[batch, max_enc_length, char_embedding]
        # attention_value:[batch, max_dec_length, char_embedding]
        attention_value = K.batch_dot(attention_mat, value, [2, 1])
        # 将各步结果拼接, attention_value => C_i, query => y_(i-1), max_pooling_over_time_extend: C_fixed(即引处Context为固定,不随时间i变化)
        # concat:[batch, max_dec_length, 3*char_embedding]
        concat = K.concatenate([attention_value, query, max_pooling_over_time_extend], axis=2)
        return concat

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1],
                input_shape[0][2]+input_shape[1][2]*2)

"""
 y: [batch, max_dec_length, char_embedding]
 x: [batch, max_enc_length, char_embedding] 
 x_mask: [batch, max_enc_length, 1]
"""
# xy: [batch, max_dec_length, 3*char_embedding], xy: C_i, y_(i-1), C_fixed
xy = Interact()([y_lstm2, x_bi_lstm2, x_mask])
# xy: [batch, max_dec_length, 3*char_embedding]
xy_dense1 = Dense(units=512, activation='relu')(xy)
xy_dense2 = Dense(units=VOCAB_SIZE_WITH_SPECIAL)(xy_dense1)
# x_dense2: [batch, max_dec_length, vocab]
# x_prior: [batch, 1, vocab_size]
# xy_average: [batch, max_dec_length, vocab]
xy_average = Lambda(lambda x: (x[0]+x[1])/2)([xy_dense2, x_prior]) # 与先验结果平均
# xy_softmax: [batch, max_dec_length, vocab]
xy_softmax = Activation(activation='softmax')(xy_average)

# 交叉熵作为loss，但mask掉padding部分
# 用 0-(t-1) 预测 1-(t)的值
# y_in:[batch, max_dec_length]
# xy_softmax:[batch, max_dec_length, vocab]
# cross_entropy:[batch, max_dec_length]
cross_entropy = K.sparse_categorical_crossentropy(target=y_in[:, 1:], output=xy_softmax[:, :-1]) # 这个loss类似于tf.seq2seq函数
# y_mask_no_start: [batch, max_dec_length, 1]
y_mask_no_start = y_mask[:,1:, 0] # y_mask中去掉开始
# loss:scalar, 计算batch中所有样本所有时间步的交叉熵loss
loss = K.sum(cross_entropy * y_mask_no_start, axis=None) / K.sum(y_mask_no_start, axis=None)
model = Model(inputs=[x_in, y_in], outputs=xy_softmax) # 此处定义input,output是为了后面预测时使用
model.add_loss(losses=loss)
model.compile(optimizer=Adam(1e-3))

def gen_title(input_string, topk=3):
    """beam search解码,
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    # xid:[topk, max_enc_length], yid:[topk,1]
    xid = np.array([str2id(input_string)] * topk) # 输入转id,将输入的句子复制3份
    yid = np.array([[START]] * topk) # 解码均以<start>开通，这里<start>的id为2
    scores = [0] * topk # 候选答案分数
    valid_start_char_id = SPECIAL_TOKEN_LEN - 1
    for time_step in range(50): # 强制要求标题不超过50字
        # xid:[topk, max_enc_length]
        # yid:[topk, time_step_length]
        # pred_proba: [topk, time_step_length, vocab_with_spec],当time_step>0时,yid的值已经用上一次的标签值来计算
        pred_proba = model.predict([xid, yid])
        # cur_proba: [topk, vocab],注意变成2维数组
        cur_proba = pred_proba[:, time_step, valid_start_char_id:] # 直接忽略<padding>、<unk>、<start>
        log_proba = np.log(cur_proba + 1e-6) # 取对数，方便计算
        # log_proba: [topk, vocab], 每行表示top_i的序列下次选择各vocab的概率
        # arg_topK:[topk, topk], 每行表示top_i的序列下次选择各top_j的概率
        # 即对于每个样本,都选出topK个预测概率最大的那几个字符index, 这个index就是charId, 即每一行是每个样本下一个字符最可能出现的topk个字符
        arg_topk = log_proba.argsort(axis=1)[:, -topk:] # argsort在axis=1上按照升序排列,此time_step中选出topk的字符
        _yid = [] # 暂存的候选目标序列, [topk, time_step_length]
        _scores = [] # 暂存的候选目标序列得分, [topk]
        if time_step == 0: # 对于time_step=0,只有1*topk种
            last_topk = 1
        else:
            last_topk = topk
        # 遍历topk*topk的组合
        for top_i in range(last_topk):
            for top_j in range(topk):
                top_i_2_j_char_index = arg_topk[top_i][top_j]
                top_i_2_j_char_id = arg_topk[top_i][top_j] + valid_start_char_id
                transfer_prob_i_2_j = log_proba[top_i][top_i_2_j_char_index] # 条件转移概率
                last_top_list = yid[top_i,:].tolist()
                _yid.append(last_top_list + [top_i_2_j_char_id])
                _scores.append(scores[top_i] + transfer_prob_i_2_j) # 因为是log域内,所以是相乘
        # 从topk*topk中选出新的topk个预测序列,注意:升序排列,即概率最大的排在最后一位
        _arg_topk = np.argsort(_scores)[-topk:]
        _yid = [_yid[k] for k in _arg_topk]
        _scores = [_scores[k] for k in _arg_topk]

        """
        if time_step == 0: # 对于time_step=0,只有1*topk种
            for top_j in range(topk):
                _yid.append(list(yid[top_j]) + [arg_topk[0][top_j] + valid_start_char_id])
                _scores.append(scores[top_j] + log_proba[0][arg_topk[0][top_j]])
        else:
            for top_i in range(topk):
                for top_j in range(topk): # 遍历topk*topk的组合
                    _yid.append(list(yid[top_i]) + [arg_topk[top_i][top_j] + valid_start_char_id]) # 在前面y的
                    _scores.append(scores[top_i] + log_proba[top_i][arg_topk[top_i][top_j]])
            _arg_topk = np.argsort(_scores)[-topk:] # 从topk*topk中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        """

        """
        yid的值每次都会变长,
        time_step=0, yid=[START]
        time_step=1, yid=[START,y0] # 假设time_step中y的预测为y0
        time_step=2, yid=[START,y0, y1] 
        time_step=3, yid=[START,y0, y1, y2] 
        ...
        """
        # 检查是否输出 <end>,若是则直接返回
        yid = []
        scores = []
        for top_j in range(topk):
            if _yid[top_j][-1] == END: # 找到<end>就返回
                return id2str(_yid[top_j])
            else:
                yid.append(_yid[top_j])
                scores.append(_scores[top_j])
        yid = np.array(yid)
    # 如果50字都找不到<end>，直接返回
    return id2str(yid[np.argmax(scores)])


s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医。'
s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午，华住集团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'

class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 训练过程中观察一两个例子，显示标题质量提高的过程
        print(gen_title(s1))
        #print(gen_title(s2))
        # 保存最优结果
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')


evaluator = Evaluate()

model.fit_generator(data_generator(),
                    steps_per_epoch=1,
                    epochs=epochs,
                    callbacks=[evaluator],
                    )