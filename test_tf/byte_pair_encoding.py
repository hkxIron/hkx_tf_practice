import re, collections

# 字节对编码算法（Byte Pair Encoding (BPE)），大致思想就是利用单个未使用的符号迭代地替换给定数据集中最频繁的符号对（原始字节）。
# 这样处理后的词表就对语言的种类不敏感了，更多关注的是语言的组织结构。

def get_stats(vocab):
   pairs = collections.defaultdict(int)
   for word, freq in vocab.items():
       symbols = word.split()
       for i in range(len(symbols)-1):
           pairs[symbols[i],symbols[i+1]] += freq
   return pairs

def merge_vocab(pair, v_in):
   v_out = {}
   bigram = re.escape(' '.join(pair))
   p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
   for word in v_in:
       w_out = p.sub(''.join(pair), word)
       v_out[w_out] = v_in[word]
   return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2, 'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 6

for i in range(num_merges):
   pairs = get_stats(vocab)
   best = max(pairs, key=pairs.get)
   vocab = merge_vocab(best, vocab)
   print(best)

"""
输出:
('e', 's')
('es', 't')
('est', '</w>')
('l', 'o')
('lo', 'w')
('n', 'e')
"""
