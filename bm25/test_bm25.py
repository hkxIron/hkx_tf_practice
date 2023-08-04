import math
from typing import *
import rank_bm25


# import jieba


# paper:http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
class BM25(object):
    def __init__(self, docs: List[List[str]]):
        """
        :param docs: 分好词的list
        """
        self.D = len(docs) # 文档数
        # 所有文档的平均长度
        self.avgdl = sum([len(doc) + 0.0 for doc in docs]) / self.D
        self.docs = docs
        self.f = []  # 每篇文档中所有词频的统计
        self.df = {} # 存储每个词所出现的文档数量
        self.idf = {}
        self.k1 = 1.5 # 参数
        self.b = 0.75 # 参数
        self.init()

    def init(self):
        for doc in self.docs:
            # 统计每一行单词出现的个数
            word_count = {}
            for word in doc:
                word_count[word] = word_count.get(word, 0) + 1 #  # 存储每个文档中每个词的出现次数
            self.f.append(word_count)

            for k, v in word_count.items():
                self.df[k] = self.df.get(k, 0) + 1 # 该词出现的文档数加1
        # 计算词的逆文档词频
        epsilon = 0.25
        for k, v in self.df.items():
            #self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5) # Robertson-Sparck Jones IDF为另一种计算方式
            #self.idf[k] = math.log(self.D + 1) - math.log(v + 0.5) # 逆文档词频 = log(词/每个词出现的文档数)
            self.idf[k] = math.log(self.D) - math.log(v + epsilon) # 逆文档词频 = log(词/每个词出现的文档数)

    def sim(self, query:List[str], index:int):
        """
        :param query: 问题
        :param index: 训练数据的下标
        :return:
        """
        score = 0
        for word in query:
            if word not in self.f[index]:
                continue
            word_cnt = self.f[index]
            d_len = len(self.docs[index])
            # The ATIRE variant of BM25
            score += self.idf[word] * word_cnt[word] * (self.k1 + 1) / (word_cnt[word] + self.k1 * (1 - self.b + self.b * d_len / self.avgdl))
        return score

    def simall(self, query:List[str]):
        """
        找出训练数据中所有相似的句子概率
        :param doc:  一句话的分词list
        :return:
        """
        scores = []
        for index in range(self.D):
            score = self.sim(query, index)
            scores.append(score)
        return scores


if __name__ == '__main__':
    corpus = [
        ['打开', '空调', '打开', '灯'],
        ['关闭', '空调', '打开', '灯'],
        ['空调', '温度', '调高', '灯', '关闭'],
    ]
    bm = BM25(corpus)
    # 统计每个小列表中出现的次数
    print(bm.f)
    # 每个单词在文档库中出现了在几个文档中
    print(bm.df)
    print(bm.idf)
    scores = bm.simall(['打开', '空调'])
    print(scores)

    #
    bm25 = rank_bm25.BM25Okapi(corpus)
    print(bm25.get_scores(['打开', '空调']))
