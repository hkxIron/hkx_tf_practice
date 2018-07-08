# tensorflow中的viterbi解码代码
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/crf/python/ops/crf.py
import numpy as np

"""
注意:此函数中的概率均已取log, 即log(p)
不知为何没有发射概率
"""
def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of TensorFlow.
  This should only be used at test time.
  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.
  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
        indices.
    viterbi_score: A float containing the score for the Viterbi sequence.
  """
  trellis = np.zeros_like(score) # size: [seq_len, num_tags], 每行标识各个隐状态的最大概率
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0] # 初始隐状态概率,score的第0行为初始隐状态概率

  for t in range(1, score.shape[0]):
    v = np.expand_dims(trellis[t - 1], 1) + transition_params
    trellis[t] = score[t] + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score

#

