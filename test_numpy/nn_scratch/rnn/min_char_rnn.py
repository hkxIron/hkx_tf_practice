"""
blog:http://karpathy.github.io/2015/05/21/rnn-effectiveness/
code: https://gist.github.com/karpathy/d4dee566867f8291f086
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
#coding:utf-8
import numpy as np
from six import *
from random import uniform

# data I/O
data = open('input1.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size)) #64,20
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1


# model parameters
# h(t) = tanh(Whx * x + Whh * h(t-1) + bh)
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden, [hidden_size, vocab_size]
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden, [hidden_size, hidden_size]
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output, [vocab_size, hidden_size]
bh = np.zeros((hidden_size, 1)) # hidden bias, [hidden_size, 1]
by = np.zeros((vocab_size, 1)) # output bias, [vocab_size, 1]

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is [H,1] array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  inputs:[seq_length]
  targets:[seq_length]
  hprev:[hidden_size, batch_size=1]
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0

  """ 
  forward:
  h(t) = tanh(Whx * x(t) + Whh*h(t-1) + bh)
  y = Why * h + by
  p = softmax(y)
  loss = - sum_k{ label_k*log(p_k) }
  """
  # forward pass
  batch_size = 1
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size, batch_size)) # encode in 1-of-k representation, xs:[seq_length, vocab_size, batch_size=1]
    xs[t][inputs[t]] = 1 # xs:[seq_length, vocab_size, batch_size=1]
    # Wxh:[hidden_size, vocab_size]
    # xs[t]:[vocab_size, batch_size=1]
    # Whh:[hidden_size, hidden_size]
    # hs[t-1]:[hidden_size, batch_size=1]
    # bh:[hidden_size, 1]
    # hs[t]:[hidden_size, batch_size=1]
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    # Why:[vocab_size, hidden_size]
    # hs[t]:[hidden_size, batch_size=1]
    # by:[vocab_size, batch_size=1]
    # ys[t]:[vocab_size, batch_size=1]
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    # ps[t]:[vocab_size, batch_size=1]
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    # ps[t]:[vocab_size, batch_size=1]
    # targets:[seq_length]
    # 只取对应的target的char的概率求loss
    # 注意:此处的loss是将整个seq的所有loss正序相加
    loss += -np.log(ps[t][targets[t], 0]) # softmax (cross-entropy loss), 此处的0,只取了embed第一维

  """
             Loss
           /  |  \
          /   |   \
        L1    L2   L3 ...   
        | Whh | Whh |
  h0 -> h1 -> h2 -> h3 ...
        |     |     |
        x1    x2    x3 ...
       因此可以看到,前向转播时,h1的信息流动: h1->h2->h3 
       因此求导数时,dh2中应有dh3,dh1中应有dh2
  """
  # backward pass: compute gradients going backwards
  # dWxh:[hidden_size, vocab_size]
  # dWhh:[hidden_size, hidden_size]
  # dWhy:[vocab_size, hidden_size]
  # dbh:[hidden_size, 1]
  # dby:[vocab_size, 1]
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  # hs[t]:[hidden_size, batch_size=1]
  # dhnext:[hidden_size, batch_size=1]
  dhnext = np.zeros_like(hs[0]) # h(T+1)初始化为0
  # 注意:此处是将inputs逆着输入
  # x10, x9, x8, x7, ...
  for t in reversed(range(len(inputs))):
    # ps[t]:[vocab_size, batch_size=1]
    # dy: [vocab_size, batch_size=1]
    dy = np.copy(ps[t]) # dy=prob
    # dL/dprob_k = prob-label
    # 只有yi=k时,label=1,其它情况下均为0
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    # y即为softmax的输入
    # loss = -sum_{t} L_t, 其中t为时间步
    # L_t = -sum_{k}{label_{k}*log(p_{k})}, 其中t为时间步,k为标签类别
    # 对于第t个时间步
    # p = softmax(y)
    # y = Why*h+by
    # h(t) = tanh(Whx * x(t) + Whh * h(t - 1) + bh)
    # =>
    #   dL/dWhy = dL/dy*h,
    #   dL/dby = dL/dy
    #   dL/dh = dL/dy*Why
    #
    # 由于 h(t+1) = tanh(Whx * x(t) + Whh * h(t) + bh), 即ht对于t+1的时间步的loss也有影响,
    # 因此t+1的梯度也需要加到ht上,即下一时间步对ht的梯度:
    #   dhnext= dL/dh(t)
    #         = dL/dh(t+1)* dtanh* Whh
    #         = dhraw *Whh
    # 注意:对于最后一个时间步T, 因此dhnext=0
    # 当前时间步对ht的梯度:dL/dh = dL/dy * Why
    # 因此,总dL/dh梯度: dL/dh = dL/dy*Why + dhnext
    #
    # dy:[vocab_size, batch_size=1]
    # hs[t]:[hidden_size, batch_size=1]
    # dWhy:[vocab_size, hidden_size]
    dWhy += np.dot(dy, hs[t].T)
    # dby:[vocab_size, 1]
    dby += dy
    # Why:[vocab_size, hidden_size]
    # dy:[vocab_size, batch_size=1]
    # dhnext:[hidden_size, batch_size=1]
    # dh:[hidden_size, batch_size=1]
    # 正向时,信息从t-1流向t, 因此求梯度时, dt会影响dt-1的值
    dh = np.dot(Why.T, dy) + dhnext # backprop into h, 这里注意:加上h(t+1)的梯度,第一次时初始化为0
    #dh = np.dot(Why.T, dy) # no dh,no influence after remove dhnext
    # h(t) = tanh(Whx * x + Whh * h(t-1) + bh) => h = tanh(dhraw), dhraw = Whx * x + Whh * h + bh
    #    dL/dhraw = dL/dh*(1-tanh(hraw)^2)
    #    dL/dbh = dL/dhraw
    #    dL/Whx = dL/dhraw*x
    #    dL/dWhh = dL/dhraw*h(t-1)
    #    dL/dh(t-1) = dL/dhraw*Whh
    # dh:[hidden_size, batch_size=1]
    # hs[t]: [hidden_size, batch_size=1]
    # dhraw: [hidden_size, batch_size=1]
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    # dbh: [hidden_size, 1]
    dbh += dhraw
    # dhraw: [hidden_size, batch_size=1]
    # xs[t]:[vocab_size, batch_size=1]
    # dWxh:[hidden_size, vocab_size]
    dWxh += np.dot(dhraw, xs[t].T)
    # dhraw: [hidden_size, batch_size=1]
    # hs[t-1]: [hidden_size, batch_size=1]
    # dWhh:[hidden_size, hidden_size]
    dWhh += np.dot(dhraw, hs[t-1].T)
    # Whh:[hidden_size, hidden_size]
    # dhraw:[hidden_size, batch_size=1]
    # dhnext = [hidden_size, batch_size=1]
    dhnext = np.dot(Whh.T, dhraw) # 注意:在最后时更新t-1的dhnext

  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

# gradient checking
def gradCheck(inputs, target, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by],
                               [dWxh, dWhh, dWhy, dbh, dby],
                               ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    assert(s0 == s1, 'Error dims dont match: %s and %s.'%(s0, s1))
    print(name)
    for i in range(num_checks):
      ri = int(uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      grad_sum_abs =  abs(grad_numerical + grad_analytic)
      if grad_sum_abs>0:
        rel_error = abs(grad_analytic - grad_numerical) / grad_sum_abs
        check_flag = rel_error <= 1e-7
        print('%f, %f => %e , check_flag:%s' % (grad_numerical, grad_analytic, rel_error, check_flag))
        # rel_error should be on order of 1e-7 or less

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  batch_size = 1
  x = np.zeros((vocab_size, batch_size))
  x[seed_ix] = 1 # 当前输入的字符char作为初始的seed
  word_indexs = []
  for t in range(n):
    # 做一次前向传播
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y)) #26*1
    word_index = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, batch_size))
    x[word_index] = 1
    word_indexs.append(word_index)
  return word_indexs

iter, start_index = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

batch_size = 1
while True:
  # 超出文本长度,则重头开始
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if start_index+seq_length+1 >= len(data) or iter == 0:
    hprev = np.zeros((hidden_size, batch_size)) # reset RNN memory, [hidden_size, batch_size=1]
    start_index = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[start_index:start_index + seq_length]] # [seq_length]
  targets = [char_to_ix[ch] for ch in data[start_index + 1:start_index + seq_length + 1]] # [seq_length]

  # sample from the model now and then
  if iter % 100 == 0:
    if iter == 0:
      gradCheck(inputs, targets[0], hprev)
    # 每100次从model中进行采样
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if iter % 100 == 0: print('iter %d, loss: %f' % (iter, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh,  Whh,  Why,  bh,  by],
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam # mem是用来记录每个变量的梯度的累积平方和
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  start_index += seq_length # move data pointer
  iter += 1 # iteration counter