from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import nltk
import numpy as np
import pickle
import random

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    #batch类，里面包含了encoder输入，decoder输入，decoder标签，decoder样本长度mask
    def __init__(self):
        self.encoder_inputs = []  #嵌套列表，每个元素都是一个句子中每个单词都id
        self.encoder_inputs_length = []  #一维列表，每个元素对应上面每个句子的长度
        self.decoder_targets = []
        self.decoder_targets_length = []

def loadDataset(filename):
    '''

    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对,里均的数据均为word_id
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))

    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']

    show_samples = []
    for sample in trainingSamples[:10]:
        question = [id2word.get(x, '<unknown>') for x in sample[0]]
        answer = [id2word.get(x, '<unknown>') for x in sample[1]]
        show_samples.append([question, answer])
    print("show samples:", show_samples)
    return word2id, id2word, trainingSamples

def createBatch(samples):
    '''
    samples: raw words
    根据给出的samples（就是一个batch的数据），进行padding并构造成placeholder所需要的数据形式
    :param samples: 一个batch的样本数据，列表，每个元素都是[question， answer]的形式，均为word id
    :return: 处理完之后可以直接传入feed_dict的数据格式
    '''
    batch = Batch()

    # 获取每个样本的长度，并保存在source_length和target_length中
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples] # question
    batch.decoder_targets_length = [len(sample[1]) for sample in samples] # answer

    # 获得一个batch样本中最大的序列长度
    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    #将每个样本进行PAD至最大长度
    for sample in samples:
        # 将source进行反序并PAD值本batch的最大长度
        # 逆序输入对模型训练有帮助
        source = list(reversed(sample[0])) # question
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source) # NOTE:由于已经逆序,所以pad加在首部

        #将target进行PAD，并添加END符号
        target = sample[1] # answer
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad) # pad加在尾部
        #batch.target_inputs.append([goToken] + target + pad[:-1])

    return batch

def getAllTrainBatches(data, batch_size):
    '''
    根据读取出来的所有数据和batch_size将原始数据分成不同的小batch。对每个batch索引的样本调用createBatch函数进行处理
    :param data: loadDataset函数读取之后的trainingSamples，就是QA对的列表,里面都是word id
    :param batch_size: batch大小
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 列表，每个元素都是一个batch的样本数据，可直接传入feed_dict进行训练
    '''

    #每个epoch之前都要进行样本的shuffle
    random.shuffle(data) # 注意:这里的data是指全部的训练数据
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches

def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :param en_de_seq_len: 列表，第一个元素表示source端序列的最大长度，第二个元素表示target端序列的最大长度
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    #分词
    #tokens = nltk.word_tokenize(sentence)
    tokens = sentence.split(" ")
    if len(tokens) > 20:
        return None
    #将每个单词转化为id
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    #调用createBatch构造batch
    batch = createBatch([[wordIds, []]])
    return batch