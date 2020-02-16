
import re, collections

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

vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
num_merges = 1000
for i in range(num_merges):
    # 获取最佳的pair=> es:9
    pairs = get_stats(vocab)
    if not pairs: # 未统计到任何pair,则退出
        break
    print("iter:", i)
    best = max(pairs, key=pairs.get)
    print("best pair:", best, " count:", pairs[best])
    # vocab:{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    # iter1: vocab_temp:{'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
    vocab_temp = merge_vocab(best, vocab)
    print("input vocab:", vocab)
    print("output vocab:", vocab_temp)
    vocab = vocab_temp

print("final vocab:")
print(vocab)