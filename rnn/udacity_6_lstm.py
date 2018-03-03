# coding:utf-8
# udacity Assignment 6
# url:https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/6_lstm.ipynb
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os,sys
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

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

#file_path="D:\\hkx\\linuxML\\nlp_dataset\\text8.zip"
file_path="D:\\tencent\\tensorflow\\tensorflow-models-master\\models-master\\tutorials\\embedding\\text8.zip"
filename = maybe_download(file_path, 31344016)


def read_data(filename):
  with zipfile.ZipFile(filename) as f:
    name = f.namelist()[0]
    data = tf.compat.as_str(f.read(name))
  return data

text = read_data(filename)
print('Data size %d' % len(text))


valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' ', 即为27
first_letter = ord(string.ascii_lowercase[0]) # a:97，即获取字母对应的10进制数字

"""
将字母变为相对于a的偏移量，其中空格为0，a为1
"""
def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0

"""
将偏移转换为字母,如0转为空格，1转为
"""
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0)) # 1->a,26->z,0->' '

# Function to generate a training batch for the LSTM model.
batch_size = 64
num_unrollings = 10 # time_step

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size # 每个batch是一段，为一个segment
    print("segment:",segment)
    self._cursor = [offset * segment for offset in range(batch_size)] # 每个batch的开始指针,有batch_size 个开始指针
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float) # batch:[batch, vocab_size]
    for b in range(self._batch_size): # 64
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch # [batch,vocab_size],即一个batch 里的各个字母其实并不相邻

  def next_batch_list(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    一个batch里的各个字母其实并不相邻，而相邻batch里同位置上的元素才相邻.
    这里产生样本的方法比较巧妙，这样训练语料才不会重复，相当于将text分成batch份，每份文本长度为segment
    每次产生一个batch时，各个segment出队第一个元素
    """
    batche_list = [self._last_batch]
    for step in range(self._num_unrollings): # time_step: 10
      batche_list.append(self._next_batch())
    self._last_batch = batche_list[-1]
    return batche_list # 每次会返回 num_unrollings+1 个batch

def characters(probabilities): # prob:ndarray,[batch,vocab]
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)] # 返回的就是一些char,['n','h','l',' ',...]

def batches2string(batches): # batches长度num_unrollings+1,每个batch:[batch,vocab]
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0] # batch个 ''
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s # 相邻batch之间，相同位置上的字符是连续的,即一个单词的不同字母分散在前后几个batch中

print("batch size: ",batch_size)
train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print("train_batches: ",batches2string(train_batches.next_batch_list()))
print("train_batches: ",batches2string(train_batches.next_batch_list()))
print("valid_batches: ",batches2string(valid_batches.next_batch_list()))
print("valid_batches: ",batches2string(valid_batches.next_batch_list()))

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0] # 点乘

"""
此处的 distribution 为概率分布, 其和为1
"""
def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

# prediction: [vocab, 1]
def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0 # 按概率分布采样一个元素
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size]) # 注意：此处b的概率之和为1
  return b/np.sum(b, 1)[:,None] # [vocab,1]

hidden_nodes = 80 # num_nodes = hidden_size
graph = tf.Graph()
with graph.as_default():
  # Parameters:这4个变量可以统一成一个大矩阵
  # Input gate: input(xt), previous output(h(t-1)), and bias.
  Wix = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_nodes], -0.1, 0.1)) # Wix*Xt,ix:Wix
  Wih = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1)) # Wh*h(t-1),im:Wh
  Wib = tf.Variable(tf.zeros([1, hidden_nodes])) # bias
  # Forget gate: input, previous output, and bias.
  Wfx = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_nodes], -0.1, 0.1))
  Wfh = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
  Wfb = tf.Variable(tf.zeros([1, hidden_nodes]))
  # Memory cell: input, state and bias.
  Wcx = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_nodes], -0.1, 0.1))
  Wcm = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
  Wcb = tf.Variable(tf.zeros([1, hidden_nodes]))
  # Output gate: input, previous output, and bias.
  Wox = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_nodes], -0.1, 0.1))
  Woh = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
  Wob = tf.Variable(tf.zeros([1, hidden_nodes]))
  # Variables saving state across unrollings.初始时，上一次的状态均置为0
  saved_output = tf.Variable(tf.zeros([batch_size, hidden_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, hidden_nodes]), trainable=False)
  # Classifier weights and biases.
  classify_weight = tf.Variable(tf.truncated_normal([hidden_nodes, vocabulary_size], -0.1, 0.1))
  classify_bias = tf.Variable(tf.zeros([vocabulary_size]))

  # Definition of the cell computation.
  # 不使用tf内置的lstm接口，而是自行定义
  def lstm_cell(input_x, last_hidden, last_cell_state): # i: input_x, last_hidden: h(t-1)
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""
    # input_x:[batch,vocab_size] Wix:[vocab_size,hidden] last_hidden:[batch,hidden] Wih:[hidden,hidden] Wib:[1,hidden]
    input_gate = tf.sigmoid(tf.matmul(input_x, Wix) + tf.matmul(last_hidden, Wih) + Wib) # [batch,hidden]
    forget_gate = tf.sigmoid(tf.matmul(input_x, Wfx) + tf.matmul(last_hidden, Wfh) + Wfb) # [batch,hidden]
    output_gate = tf.sigmoid(tf.matmul(input_x, Wox) + tf.matmul(last_hidden, Woh) + Wob) # [batch,hidden]
    cell_state_candidate = tf.matmul(input_x, Wcx) + tf.matmul(last_hidden, Wcm) + Wcb # cell_state_candidate, [batch,hidden]
    last_cell_state = forget_gate * last_cell_state + input_gate * tf.tanh(cell_state_candidate) # cell_state, [batch,hidden]
    hidden_state = output_gate * tf.tanh(last_cell_state) # hidden_state,shape:[batch*hidden], [batch,hidden]
    #print("input_state:",input_gate," hidden_state:",hidden_state)
    return hidden_state, last_cell_state # [batch,hidden]

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1): # 定义多个placeholder,每个ploceholder作为一个batch的输入
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
  # 从数据可以看出，明显是训练字符级别的语言模型
  train_inputs = train_data[:num_unrollings] # list:长度time_step,每个里面都是 [batch*vocab_size]
  train_labels = train_data[1:]  # list:长度也为time_step, labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  last_hidden = saved_output # [batch, hidden]
  last_cell_state = saved_state # [batch, hidden]
  for input_index in train_inputs:# num_unrollings个[batch,vocab]，将train_inputs里的每个place_holder作为一个time_step的输入,同时将上一次的输出也作为输入
    last_hidden, last_cell_state = lstm_cell(input_index, last_hidden, last_cell_state) # input_x:[batch,vocab_size] last_hidden:[batch,hidden] last_cell_state:[batch,hidden]
    outputs.append(last_hidden) # 将time_step个output连接起来, last_hidden:[batch,hidden]

  # State saving across unrollings.
  # tf.control_dependencies()设计是用来控制计算流图的，给图中的某些计算指定顺序
  # 将last_hidden的值赋给saved_output, last_cell_state 赋给 saved_state
  # saved_output的值最初为0，现在再将更新后的last_hidden的值赋给saved_output(但个人觉得这个值目前好像并没有什么用处)
  with tf.control_dependencies([saved_output.assign(last_hidden), saved_state.assign(last_cell_state)]):
    # Classifier.
    # 将时间展开后的隐层hidden_state连接起来,然后进行分类
    # outputs: 共num_unrollings个[batch,hidden], time_step: num_unrollings
    lstm_hidden_seq = tf.concat(outputs, axis=0) # [batch*time_step,hidden],启示我们list也是可以 concat
    label_seq = tf.concat(train_labels, axis=0)
    logits = tf.nn.xw_plus_b(lstm_hidden_seq, classify_weight, classify_bias) # 此处并未取log, classify_weight:[hidden,vocab_size]
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_seq, logits=logits))
    print("logits: ",logits, " lstm_hidden_seq: ",lstm_hidden_seq) # logits: 640*27,即 [batch*time_step, vocab_size]

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, vars = zip(*optimizer.compute_gradients(loss)) # 优化loss
  gradients, _ = tf.clip_by_global_norm(gradients,clip_norm= 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, vars), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits) # [batch*time_step, vocab_size]
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, hidden_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, hidden_nodes]))
  reset_sample_state = tf.group( # 将上一次的状态重置
    saved_sample_output.assign(tf.zeros([1, hidden_nodes])),
    saved_sample_state.assign(tf.zeros([1, hidden_nodes])))
  sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output), # sample_output -> saved_sample_output
                                saved_sample_state.assign(sample_state)]): # sample_state -> saved_sample_state
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, classify_weight, classify_bias))

# -------------------------
num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  mean_loss = 0
  for step in range(num_steps):
    train_batches_data = train_batches.next_batch_list() # num_unrollings个 [batch,vocab]
    feed_dict = dict()
    for input_index in range(num_unrollings + 1): # 10+1=11
      # train_data[input_index]是place_holder
      feed_dict[train_data[input_index]] = train_batches_data[input_index] #将数据填充到place_holder
    _, batch_loss, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += batch_loss
    if step % summary_frequency == 0:
      if step > 0: mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(train_batches_data)[1:]) # train_batches_data本来就是list
      # perdictions: [batch*time_step, vocab_size] labels:[batch*time_step, vocab_size]
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels)))) # exp(entropy)
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          # 第一次随机采样一个字母
          sample_one_hot = sample(random_distribution())
          sentence = characters(sample_one_hot)[0] # 只返回一个字母
          reset_sample_state.run() #  重置上一次的状态
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: sample_one_hot})
            sample_one_hot = sample(prediction) # 利用lstm预测的概率来输出下一个字母
            sentence += characters(sample_one_hot)[0]
          print("sentence: ",sentence)
        print('=' * 80)

      # Measure validation set perplexity.
      reset_sample_state.run() # 清空上一次的hidden_state
      valid_logprob = 0
      for _ in range(valid_size):
        validate_batch = valid_batches.next_batch_list() # 每次会产生 unrollings+1个batch
        predictions = sample_prediction.eval({sample_input: validate_batch[0]})
        valid_logprob += logprob(predictions, validate_batch[1]) # 交叉熵
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size))) # 混杂度
