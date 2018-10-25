from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#语料库
corpus = [
    'This is the first document', # 0
    'This is the second document', # 1
    'welcome to my company',  # 2
    'welcome to china', # 3
    'where are you from' # 4
]

words = CountVectorizer().fit_transform(corpus)
print("words count:", words)
tfidf_stat = TfidfTransformer().fit_transform(words)
print("tfidf:", tfidf_stat)


# 手动求解tfidf
import math
def tf(word:str, count:dict):
    return count[word] / sum(count.values()) # 词w在当前文档中的出现次数, 该文档中所有词的出现总次数

def n_containing(word:str, count_list):
    return sum(1 for count in count_list if word in count)

def idf(word:str, count_list): # 文档总数, 包含该词的文档数
    return math.log(len(count_list) / (1 + n_containing(word, count_list)))

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)

from collections import defaultdict
def getCountDict(in_word_list):
   count_dt = {}
   for x in in_word_list:
       count_dt[x] = count_dt.setdefault(x, 0) + 1
   return count_dt

# 测试：
doc_count_list = [ getCountDict(x.split(" ")) for x in corpus]
for i, doc_count in enumerate(doc_count_list):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, doc_count, doc_count_list) for word in doc_count}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:5]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
