import os
import re
import zipfile
import itertools
import threading
import numpy as np
from tqdm import trange, tqdm
from collections import namedtuple
import tensorflow as tf

"""
每行是一条数据，由于一条太长，所以分了三行显示。输入和target由output隔开，每个输入的点由两个坐标构成。
假设有n个点, 前n个(x,y)的pair, output之后为n个index
x1             y1             x2             y2              x3             y3             x4            y4             x5             y5              x6              y6
0.607122483376 0.664447268879 0.953592710256 0.0215187266035 0.757626025721 0.921024039084 0.58637621508 0.433565269284 0.786836511244 0.0529589389174 0.0160877248199 0.581436054061 0.496714219523 0.633570685486 0.227776956853 0.971433036801 0.665490132665 0.074330503455 0.38355557137 0.10439215522 output 1 3 8 6 2 5 9 10 4 7 1 
0.93053373497 0.747036450998 0.277411711099 0.93825232871 0.79459230592 0.794284772785 0.96194634906 0.261223286824 0.0707955411585 0.384301925429 0.0970348242202 0.796305967116 0.452332110479 0.412415030566 0.341412603409 0.566108471934 0.247171696984 0.890328553326 0.42997841152 0.232969556152 output 1 3 2 9 6 5 8 7 10 4 1 
0.686711879502 0.0879416814813 0.443054163982 0.277818042302 0.494768607889 0.985289269001 0.559705861867 0.861138032601 0.532883570753 0.351912899644 0.712560683115 0.199273065174 0.554681363071 0.657214249691 0.90998623012 0.277140700191 0.931064195448 0.639287329779 0.398927025212 0.406909068041 output 1 6 8 9 3 4 7 10 2 5 1 
"""

def read_paper_dataset(path):
    enc_seq , target_seq , enc_seq_length , target_seq_length = [],[],[],[]
    tf.logging.info("Read dataset {} which is used in the paper..".format(path))
    length = max(re.findall('\d+', path))
    with open(path,'r') as f:
        for l in tqdm(f):
            # 使用output分割数据
            inputs, outputs = l.split(' output ') # inputs:由点的(x,y)组成的数组pair, output:为点序列组成的index
            inputs = np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]) #
            outputs = np.array(outputs.split(), dtype=np.int32)[:-1] # y最后一个1不取
            # 分割成横纵坐标两列
            enc_seq.append(inputs) # [n, 2], 第一列x,第二列y
            target_seq.append(outputs)  # skip the last one
            enc_seq_length.append(inputs.shape[0]) # 每个样本里点序列的长度
            target_seq_length.append(outputs.shape[0])
    return enc_seq,target_seq,enc_seq_length,target_seq_length


def gen_data(path):
    input_xy, output_index, enc_seq_length, target_seq_length = read_paper_dataset(path)
    # max_length 是 config的序列的最大长度
    enc_seq = np.zeros([len(input_xy), 10, 2], dtype=np.float32) # [batch, max_encoder_length, dim=2],开始时全以0填充
    target_seq = np.zeros([len(output_index), 10], dtype=np.int32) # [batch, max_decoder_length]

    # 这里的作用就是将所有的输入都变成同样长度，用0补齐
    for idx, (input_xy_array, target_index) in enumerate(tqdm(zip(input_xy, output_index))):
        enc_seq[idx, :len(input_xy_array)] = input_xy_array # 将第idx个样本中有效的xy值填入其中,其它地方保持为0
        target_seq[idx, :len(target_index)] = target_index
    return enc_seq,target_seq,enc_seq_length,target_seq_length