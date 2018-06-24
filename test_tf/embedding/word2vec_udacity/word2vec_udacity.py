# coding:utf-8
# 其实这个就是tensorflow 官网的 word2vec_basic.py
# udacity link:https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
# %matplotlib inline
from __future__ import print_function
import collections
import math
import numpy as np
import os,sys
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

file_path="D:\\hkx\\linuxML\\nlp_dataset\\text8.zip"
#file_path="D:\\tencent\\tensorflow\\tensorflow-models-master\\models-master\\tutorials\\embedding\\text8.zip"
filename = maybe_download(file_path, 31344016)


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
print('Data size %d' % len(words)) # 1700,5207

vocabulary_size = 50000

# word:list(str)
def build_dataset(words):
    # words:list,所有的句子组成了一个大的list，其长度为17005207
    print("len:",len(words)," words:",words[0:10]) # len: 17005207  words: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
    #sys.exit(-1)
    count = [['UNK', -1]] # (word,词频)
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1)) # 最常见的4999个词
    dictionary = dict()
    # print(count) # count里除了第一个元素，每个元素都是二维元组，第一个是word,第二个是词频
    # 建立word -> 索引 的映射
    for word, _ in count:
        dictionary[word] = len(dictionary) # word -> 索引 , 开始时len=0
    index_data = list()
    unk_count = 0
    # 将语料中所有的词替换成索引值
    for word in words:
        if word in dictionary:
            index = dictionary[word] # 获取索引值
        else:
            index = 0  # dictionary['UNK']，未登录词使用0代替
            unk_count = unk_count + 1
        index_data.append(index)

    count[0][1] = unk_count # 更新未登录词的统计数量
    # 将词的索引与word交换，形成索引->word的映射
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return index_data, count, dictionary, reverse_dictionary


index_data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', index_data[:10]) # data的前10个词的index
del words  # Hint to reduce memory.

data_index = 0

#skip_window:当前中心词的取词半径，如果是1，即左右各取1词
# 样本数：batch_size = 8 , 一个次被用作label的次数：num_skips=2, skip_window=1
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0 # 断定能整除
    assert num_skips <= 2 * skip_window  # 采样的个数要小于或等于窗口的大小
    batch = np.ndarray(shape=(batch_size), dtype=np.int32) # 大小为batch_size,值为随机初始化
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32) # 标签
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span) # 双端队列
    # 将元素入队
    for _ in range(span): # 2*r+1
        buffer.append(index_data[data_index])
        data_index = (data_index + 1) % len(index_data)
    # 在一个窗口内，需要采出batch_size=8个样本,而每次只能采num_skip=2个样本，因此需要重复采4次
    for i in range(batch_size // num_skips): # 8//2 = 4
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        # 每个batch内需要选出num_skips个样本
        for j in range(num_skips):
            # target就是label,是不能重复的
            while target in targets_to_avoid: # 如果采样的词是label(即target)，那么就一直随机选，直到选到其它的词
                target = random.randint(0, span - 1) # (0,1,2)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        # 每选一个batch, index++
        buffer.append(index_data[data_index])
        data_index = (data_index + 1) % len(index_data)
    return batch, labels # label里返回的都是正样本

# 获取前8个词索引对应的真正词
print('data from index:', [reverse_dictionary[di] for di in index_data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]: #
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)]) # label里都是正样本
"""
len: 17005207  words: ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against']
with num_skips = 2 and skip_window = 1:
    batch: ['originated', 'originated', 'as',        'as', 'a',   'a',  'term', 'term']
    labels: ['as',        'anarchism',  'originated', 'a', 'as', 'term', 'a',    'of']
即skip-gram正样本为：
originated->as
originated->anarchism
as -> originated
as -> a
a -> as
a -> term
term -> a
term -> of 

with num_skips = 4 and skip_window = 2:
    batch: ['as',     'as',        'as', 'as',        'a',   'a',         'a',  'a']
    labels: ['term', 'originated', 'a',  'anarchism', 'of', 'originated', 'as', 'term']
从上面的样本也可以看出，是将中心词附近的样本作为正例，既然是确定的，为何还要搞随机？
"""


batch_size = 150
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size)) # 从100个元素里采样16个,ndarray:16维
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) # 5w*128
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    # 有多少个输出，就有多少个bias,记住输出层是有bias的
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset) # batch_size*embedding_size
    print("tensor embed:",embed)
    # Compute the softmax loss, using a sample of the negative labels each time.
    # 但是google给出的word2vec代码中，计算的是标准nce loss，而不是用sampled softmax近似softmax,
    # 并且sampled_softmax仅用于训练，而在测试时使用标准的softmax
    # tf的文档上建议将partition_strategy="div"
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size,partition_strategy="div"))

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True)) # 所有词的embedding求模（先逐元素平方求和）
    normalized_embeddings = embeddings / norm # V*D, V为词汇表的大小
    print("normalized_embeddings:",normalized_embeddings) # V*D
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) # N*D, N为本次验证集的样本个数
    print("valid embeddings:",valid_embeddings)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings)) # N*D* ((V*D).T) ，验证集中的词与所有其他的词间的相似度
    print("similarity:",similarity) # N*V

num_steps = 100001
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels} # 这里的train_dataset是上面graph里的train_dataset
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        # 计算两两词的相似度
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                # 计算每个词与其embedding相似度最高的8个词
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

# 可视化一部分点
num_points = 400

# 降维可视化
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    # 给每个点加注释
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)


