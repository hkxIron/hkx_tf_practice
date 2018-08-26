import time

import numpy as np
import tensorflow as tf

# source repo: https://github.com/hkxIron/deep-learning/blob/master/embeddings/Skip-Grams-Solution.ipynb

import utils

from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import zipfile

dataset_folder_path = 'D:\\hkx\\linuxML\\nlp_dataset\\' # 数据解压路径
dataset_filename = 'D:\\hkx\\linuxML\\nlp_dataset\\text8.zip'
dataset_name = 'Text8 Dataset'

# 用来显示下载进度条
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


if not isfile(dataset_filename):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc=dataset_name) as pbar:
        urlretrieve(
            'http://mattmahoney.net/dc/text8.zip',
            dataset_filename,
            pbar.hook)

if not isdir(dataset_folder_path):
    with zipfile.ZipFile(dataset_filename) as zip_ref:
        zip_ref.extractall(dataset_folder_path) # 解压

import os
with open(os.path.join(dataset_folder_path,'text8')) as f:
    text = f.read()

# words为文本中的所有单词序列
words = utils.preprocess(text) # 大写转小写,以及符号替换,去掉低频词
print(words[:30])
print("Total words: {}".format(len(words))) #16680599
print("Unique words: {}".format(len(set(words)))) #63641
vocab_to_int, int_to_vocab = utils.create_lookup_tables(words) # word ->index , index -> word
int_words = [vocab_to_int[word] for word in words]

from collections import Counter
import random

# 计算丢弃概率,与其出现频率正相关
# 注意:丢词是针对文本里的所有单词而言,而非针对某个窗口
threshold = 1e-5
word_counts = Counter(int_words)
total_count = len(int_words)
freqs = {word: count/total_count for word, count in word_counts.items()}
p_drop = {word: 1 - np.sqrt(threshold/freqs[word]) for word in word_counts}
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

def get_target(words, idx, window_size=5):
    ''' Get a list of words in a window around an index. '''
    R = np.random.randint(low=1, high=window_size + 1) #
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = set(words[start:idx] + words[idx + 1:stop + 1]) # 不包含当前词,用idx预测周围的词
    return list(target_words)

def get_batches(words, batch_size, window_size=5):
    ''' Create a generator of word batches as a tuple (inputs, targets) '''
    n_batches = len(words) // batch_size
    # only full batches
    words = words[:n_batches * batch_size]
    for idx in range(0, len(words), batch_size): # 步长为batch_size
        x, y = [], []
        # the dog jump over the cat
        batch = words[idx:idx + batch_size] #当前batch
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            x.extend([batch_x] * len(batch_y)) # dog
            y.extend(batch_y) #  the, jump, over,...
        yield x, y # generator

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32, [None], name='inputs') # dog
    labels = tf.placeholder(tf.int32, [None, None], name='labels') #  the, jump, over,...

n_vocab = len(int_to_vocab)
n_embedding = 200 # Number of embedding features
with train_graph.as_default():
    embedding = tf.Variable(tf.random_uniform((n_vocab, n_embedding), -1, 1)) # vocab * embed_dim
    embed = tf.nn.embedding_lookup(embedding, inputs) # input: batch, embed: batch* embed_dim

# Number of negative labels to sample
n_sampled = 100
with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal((n_vocab, n_embedding), stddev=0.1)) # W: vocab * embed_dim
    softmax_b = tf.Variable(tf.zeros(n_vocab)) # W: vocab

    # Calculate the loss using negative sampling
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                      labels, embed,
                                      n_sampled, n_vocab,
                                      partition_strategy="div"
                                      )

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with train_graph.as_default():
    ## From Thushan Ganegedara's implementation
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size // 2))

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True)) # embedding:vocab * embed_dim, norm: vocab*1
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset) # k*embed_dim
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding)) # valid_embed: k* embed_dim, normalized_embed: vocab*embed_dim

epochs = 10
batch_size = 1000
window_size = 10

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        for x, y in batches: # x:[the, the, the,...] y : [dog, jumps, over, ...], x_len == y_len
            feed = {inputs: x, labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            if iteration % 1000 == 0:
                # note that this is expensive (~20% slowdown if computed every 500 steps)
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1
    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    embed_mat = sess.run(embedding)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
viz_words = 500
tsne = TSNE()
embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :]) # 展现前面500个单词

fig, ax = plt.subplots(figsize=(14, 14))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='steelblue')
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
# 在python中需要show
plt.show()