# blog: https://www.jianshu.com/p/83443b2baf27
# git: https://github.com/princewen/tensorflow_practice/blob/master/nlp/basic_seq2seq.py
# 鸡鸣万户晓 鹤舞一年春
# 戌岁祝福万事顺 狗年兆丰五谷香
import tensorflow as tf
from tensorflow.python.layers.core import Dense

import numpy as np
import time

# 超参数
# Number of Epochs
epochs = 500
# Batch Size
batch_size = 50
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

source = open("data/source.txt", 'w', encoding="utf-8") # 上联作为 encoder的输入
target = open("data/target.txt", 'w', encoding="utf-8") # 下联作为 decoder的输出

with open("data/对联.txt", 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(" ")
        source.write(line[0] + '\n')
        target.write(line[1] + '\n')
source.close()
target.close()

with open('data/source.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

with open('data/target.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

print(source_data.split('\n')[:10])
print(target_data.split('\n')[:10])


def extract_character_vocab(data):
    """
    :param data:
    :return: 字符映射表
    """
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_words = list(set([character for line in data.split('\n') for character in line]))
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


# 得到输入和输出的字符映射表 (有必要计算两次么,不都是一样的么?)
source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data + target_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(source_data + target_data)

# 将每一行转换成字符id的二维list,每行为一个上联
# source如:承上下求索志
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>']) for letter in line]
                for line in source_data.split('\n')] # 上联进行转换

# TODO: 注意:seq2seq中, decoder末尾有一个 "<eos>"
# target如:绘春秋振兴图
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>']) for letter in line]
              + [target_letter_to_int['<EOS>']]
                for line in target_data.split('\n')]

print(source_int)
print(target_int)


# 输入层
def get_input_tensors():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs') # [batch, source_sequence_length]
    targets = tf.placeholder(tf.int32, [None, None], name='targets') # [batch, target_sequence_length]
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length') # [batch,]
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len') # [1]
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length') # [batch]
    # inputs:[batch, source_sequence_length], targets:[batch, target_sequence_length]
    # target_sequence_length:[batch,], max_target_sequence_length:[1]
    # source_sequence_length:[batch,]
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


# Encoder
"""
在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
在Embedding中，我们使用tf.contrib.layers.embed_sequence，它会对每个batch执行embedding操作。
"""


def get_encoder_layer(input_data,
                      rnn_size,
                      num_layers,
                      source_sequence_length,
                      source_vocab_size,
                      encoding_embedding_size):
    """
    构造Encoder层
    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    """
    # https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/contrib/layers/embed_sequence
    """
    embed_sequence(
    ids,
    vocab_size=None,
    embed_dim=None,
    unique=False,
    initializer=None,
    regularizer=None,
    trainable=True,
    scope=None,
    reuse=None
    )
    ids: [batch_size, doc_length] Tensor of type int32 or int64 with symbol ids.

    return : Tensor of [batch_size, doc_length, embed_dim] with embedded sequences.
    """

    # input_data:[batch, source_sequence_length]
    # encoder_input:[batch, source_sequence_length, embedding_size]
    # 这个高级接口,都不需要申请变量了
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell
    # 多层lstm
    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

    # encoder_embed_input:[batch, source_sequence_length, embedding_size]
    # source_sequence_length:[batch]

    # encoder_output:[batch, source_sequence_length, hidden_size]
    # encoder_state:[hidden=[batch, hidden_size], cell=[batch, hidden_size]]
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=cell,
                                                      inputs=encoder_embed_input,
                                                      sequence_length=source_sequence_length,
                                                      dtype=tf.float32)

    return encoder_output, encoder_state


"""
我们首先需要对target端的数据进行一步预处理。在我们将target中的序列作为输入给Decoder端的RNN时，
序列中的最后一个字母（或单词）其实是没有用的,
我们此时只看右边的Decoder端，可以看到我们的target序列是[<go>, W, X, Y, Z, <eos>]，
其中<go>，W，X，Y，Z是每个时间序列上输入给RNN的内容，我们发现，<eos>并没有作为输入传递给RNN。
因此我们需要将target中的最后一个字符去掉，同时还需要在前面添加<go>标识，告诉模型这代表一个句子的开始。

即
encoder的输入:  A,B,C
decoder里的输入:<go>,W,X,Y,Z
decoder的输出:    W ,X,Y,Z,<eos>
"""
# decoder output:W,X,Y,Z,<eos>
# => decoder input: <go>,W,X,Y,Z
def process_decoder_input(data,
        vocab_to_int,
        batch_size):
    # data:[batch, decoder_sequence_length]
    # ending:[batch, decoder_sequence_length-1]
    # batch_size = data.get_shape().as_list()[0]
    # 此处用到batch_size,因此 训练时的batch_size与serving时的必须一样才行
    # TODO:此处有误,由于已经进行了pad,所以最后一个元素不是<EOS>,而是PAD
    ending = tf.strided_slice(input_=data, begin=[0, 0], end =[batch_size, -1], strides=[1, 1]) # 每个样本中去掉最后一个字符<eos>
    # filled_value:[batch, 1]
    filled_value = tf.fill(dims=[batch_size, 1], value=vocab_to_int['<GO>']) # 每个target序列前面加上<GO>,即begin开始符
    # data:[batch, decoder_sequence_length]
    decoder_input = tf.concat([filled_value, ending], axis=1)
    return decoder_input

def decoding_layer(target_letter_to_int,
                   decoding_embedding_size,
                   num_layers,
                   rnn_size,
                   target_sequence_length,
                   max_target_sequence_length,
                   encoder_state,
                   decoder_input):
    '''
    构造Decoder层
    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''

    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    # decoder_embeddings:[vocab, embedding_size]
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    # decoder_input:[batch, decoder_input_length]
    # decoder_embed_input:[batch, decoder_input_length, embedding_size]
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell

    # 2.构建多层rnn
    multi_rnn_cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # 3.Output全连接层,用于连接decoder rnn的hidden层到vocab_size层的连接
    # output_layer:[batch, decoder_output_length, target_vocab_size], target_vocab_size定义了输出层的大小
    output_layer = Dense(units=target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1))

    # 4. Training decoder(注意:此处是训练阶段)
    with tf.variable_scope("decode"):
        # decoder_embed_input:[batch, decoder_input_length, embedding_size]
        # target_sequence_length:[batch]
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # 注意 (TODO):decoder部分中的initial_state需要以encoder中的最后一个hidden_state作为初始输入,这个是seq2seq的关键
        # encoder_state: [hidden = [batch, hidden_size], cell = [batch, hidden_size]]
        # output_layer: [target_vocab_size]
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=multi_rnn_cell, # 定义decoder里的rnn cell
                                                           helper=training_helper, # 定义decoder里的embedding输入
                                                           initial_state=encoder_state, # decoder里的hidden,cell初始化状态
                                                           output_layer=output_layer) # 输出层大小

        """
        dynamic_decode
        用于构造一个动态的decoder，返回的内容是：(final_outputs, final_state, final_sequence_lengths).
        其中，final_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
        rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
        sample_id: [batch_size, decoder_targets_length], tf.int32，保存最终的解码结果。可以表示最后的答案
        """
        # training_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size],
        #                          sample_id:[batch_size, decoder_targets_length])
        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True,
                                                                          maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder (注意,此处是预测,用greedySearch或者beamSearch)
    # 与training共享参数
    # encoder的输入:  A,B,C
    # decoder里的输入:<go>,W,X,Y,Z
    # decoder的输出:    W ,X,Y,Z,<eos>
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        # start_tokens:[batch_size,],内容均为:'<GO>'
        start_tokens = tf.tile(input=tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32),
                               multiples=[batch_size],
                               name='start_token')

        # predict时用greedySearch
        # decoder_embeddings:[vocab, embedding_size]
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=decoder_embeddings,
                                                                     start_tokens=start_tokens,
                                                                     end_token=target_letter_to_int['<EOS>'])

        # encoder_state: [hidden = [batch, hidden_size], cell = [batch, hidden_size]]
        # output_layer:[batch, decoder_output_length, target_vocab_size]
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell=multi_rnn_cell,
                                                             helper=predicting_helper,
                                                             initial_state=encoder_state,
                                                             output_layer=output_layer)
        # predicting_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size],
        #                            sample_id:[batch_size, decoder_targets_length])
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                                            impute_finished=True,
                                                                            maximum_iterations=max_target_sequence_length)

    # training_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size], sample_id:[batch_size, decoder_targets_length ])
    # predicting_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size], sample_id:[batch_size, decoder_targets_length ])
    return training_decoder_output, predicting_decoder_output


# 上面已经构建完成Encoder和Decoder，下面将这两部分连接起来，构建seq2seq模型
def seq2seq_model(input_data,
                  targets,
                  lr,
                  target_sequence_length,
                  max_target_sequence_length,
                  source_sequence_length,
                  source_vocab_size,
                  target_vocab_size,
                  encoder_embedding_size,
                  decoder_embedding_size,
                  rnn_size, num_layers):
    # input_data:[batch, source_sequence_length], targets:[batch, target_sequence_length]
    # encoder_output:[batch, source_sequence_length, hidden_size]
    # encoder_state:[hidden=[batch, hidden_size], cell=[batch, hidden_size]]
    _, encoder_state = get_encoder_layer(input_data,
                                         rnn_size,
                                         num_layers,
                                         source_sequence_length,
                                         source_vocab_size,
                                         encoding_embedding_size)

    # 处理decoder input,即decoder output:W,X,Y,Z,<eos> => decoder input: <go>,W,X,Y,Z
    # decoder_input:[batch, decoder_sequence_length]
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    # training_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size], sample_id:[batch_size, decoder_targets_length])
    # predicting_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size], sample_id:[batch_size, decoder_targets_length])
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                        decoding_embedding_size,
                                                                        num_layers,
                                                                        rnn_size,
                                                                        target_sequence_length,
                                                                        max_target_sequence_length,
                                                                        encoder_state,
                                                                        decoder_input)

    return training_decoder_output, predicting_decoder_output


# 构造graph
train_graph = tf.Graph()

with train_graph.as_default():
    # inputs:[batch, source_sequence_length], targets:[batch, target_sequence_length]
    # target_sequence_length:[batch,], max_target_sequence_length:[1]
    # source_sequence_length:[batch,]
    input_data, targets, \
    lr, target_sequence_length, \
    max_target_sequence_length, source_sequence_length = get_input_tensors()

    # training_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size], sample_id:[batch_size, decoder_targets_length])
    # predicting_decoder_output:(rnn_output:[batch_size, decoder_targets_length, vocab_size], sample_id:[batch_size, decoder_targets_length])
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                       targets,
                                                                       lr,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       source_sequence_length,
                                                                       len(source_letter_to_int),
                                                                       len(target_letter_to_int),
                                                                       encoding_embedding_size,
                                                                       decoding_embedding_size,
                                                                       rnn_size,
                                                                       num_layers)

    # training_logits:[batch_size, decoder_targets_length, vocab_size]
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    # predicting_logits:[batch_size, decoder_targets_length]
    predicting_sample_ids = tf.identity(predicting_decoder_output.sample_id, name='predictions')

    # mask是权重的意思
    # tf.sequence_mask([1, 3, 2], 5)
    #  [[True, False, False, False, False],
    #  [True, True, True, False, False],
    #  [True, True, False, False, False]]

    # target_sequence_length:[batch, target_sequence_length]
    # target_sequence_mask:[batch, max_target_sequence_length]
    target_sequence_mask = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name="masks")

    # logits: A Tensor of shape [batch_size, sequence_length, num_decoder_symbols] and dtype float.
    # The logits correspond to the prediction across all classes at each timestep.
    # targets: A Tensor of shape [batch_size, sequence_length] and dtype int.
    # The target represents the true class at each timestep.
    # weights: A Tensor of shape [batch_size, sequence_length] and dtype float.
    # weights constitutes the weighting of each prediction in the sequence. When using weights as masking,
    # set all valid timesteps to 1 and all padded timesteps to 0, e.g. a mask returned by tf.sequence_mask.
    with tf.name_scope("optimization"):
        # training_logits:[batch_size, decoder_targets_length, vocab_size]
        # targets:[batch_size, decoder_targets_length]
        # target_sequence_mask:[batch, max_target_sequence_length]
        # cost: scalar
        cost = tf.contrib.seq2seq.sequence_loss(
            logits=training_logits,
            targets=targets,
            weights=target_sequence_mask # weights参数常常使用我们1.11中得到的mask。
        )

        optimizer = tf.train.AdamOptimizer(lr)

        # minimize函数用于添加操作节点，用于最小化loss，并更新var_list.
        # 该函数是简单的合并了compute_gradients()与apply_gradients()函数返回为一个优化更新后的var_list，
        # 如果global_step非None，该操作还会为global_step做自增操作

        # 这里将minimize拆解为了以下两个部分：

        # 对var_list中的变量计算loss的梯度 该函数为函数minimize()的第一部分，返回一个以元组(gradient, variable)组成的列表
        gradients = optimizer.compute_gradients(cost)
        clippped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        # 将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
        train_op = optimizer.apply_gradients(clippped_gradients)

def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence_length_in_batch = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence_length_in_batch - len(sentence))
            for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        # start_i:start_i+batch_size-1
        sources_batch = sources[start_i: start_i + batch_size]
        targets_batch = targets[start_i: start_i + batch_size]
        # padding
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        # 计算每个batch中各句子的真实长度
        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))
        # 计算每个batch中各句子的真实长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

# Train,从第batch_size:end-1的行作为训练数据
train_source = source_int[batch_size:]
train_target = target_int[batch_size:]

# 留出一个batch进行验证,从第0:batch_size-1的行作为测试数据
valid_source = source_int[:batch_size]
valid_target = target_int[:batch_size]

(valid_targets_batch,
 valid_sources_batch,
 valid_targets_lengths,
 valid_sources_lengths) = next(
    get_batches(valid_target,
                valid_source,
                batch_size,
                source_letter_to_int['<PAD>'],
                target_letter_to_int['<PAD>'])
)

display_step = 50
checkpoint = "data/trained_model.ckpt"

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    print()
    for epoch_i in range(1, epochs + 1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target,
                            train_source,
                            batch_size,
                            source_letter_to_int['<PAD>'],
                            target_letter_to_int['<PAD>'])
        ):
            feed_dict= {
                input_data: sources_batch,
                targets: targets_batch,
                lr: learning_rate,
                target_sequence_length: targets_lengths,
                source_sequence_length: sources_lengths
            }

            _, train_loss = sess.run([train_op, cost], feed_dict=feed_dict)

            if batch_i % display_step == 0:
                # 计算validation loss
                validation_loss = sess.run(
                    [cost],
                    {input_data: valid_sources_batch,
                     targets: valid_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: valid_targets_lengths,
                     source_sequence_length: valid_sources_lengths})

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              train_loss,
                              validation_loss[0]))
    # 训练完毕
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')

# 预测
def source_letter_to_int_seq(text):
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] \
           + [ source_letter_to_int['<PAD>']] * (sequence_length - len(text))

# 自定义一个对联,测试看看
# 用训练的模型进行预测
input_word = '戌岁祝福万事顺'
text_id = source_letter_to_int_seq(input_word)

checkpoint = "data/trained_model.ckpt"
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    predict_sample_ids = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    #batch_size=1,若设为1则会出错
    answer_sample_ids = sess.run(predict_sample_ids, {input_data: [text_id] * batch_size,
                                                      target_sequence_length: [len(input_word)] * batch_size,
                                                      source_sequence_length: [len(input_word)] * batch_size})[0]

    pad = source_letter_to_int["<PAD>"]

    print('原始输入:', input_word)
    print('\nSource')
    print('  Word 编号:    {}'.format([i for i in text_id]))
    print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text_id])))

    print('\nTarget')
    print('  Word 编号:       {}'.format([i for i in answer_sample_ids if i != pad]))
    print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_sample_ids if i != pad])))

