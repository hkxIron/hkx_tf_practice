#!/usr/bin/python
import sys
import numpy as np
import gensim

from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from sklearn.cross_validation import train_test_split

LabeledSentence = gensim.models.doc2vec.LabeledSentence
# IMDB数据集见：http://ai.stanford.edu/~amaas/data/sentiment/
# 使用Doc2Vec进行分类任务，我们使用 IMDB电影评论数据集作为分类例子，测试gensim的Doc2Vec的有效性。
# 数据集中包含25000条正向评价，25000条负面评价以及50000条未标注评价
pos_file = "pos.txt"
neg_file = "neg.txt"
unsup_file = "unsup.txt"
##读取并预处理数据
def get_dataset():
    #读取数据
    with open(pos_file,'r') as infile:
        pos_reviews = infile.readlines()
    with open(neg_file,'r') as infile:
        neg_reviews = infile.readlines()
    with open(unsup_file,'r') as infile:
        unsup_reviews = infile.readlines()

    #使用1表示正面情感，0为负面
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    #将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

    #对英文做简单的数据清洗预处理，中文根据需要进行修改
    def cleanText(corpus):
        punctuation = """.,?!:;(){}[]"""
        corpus = [z.lower().replace('\n','') for z in corpus]
        corpus = [z.replace('<br />', ' ') for z in corpus]

        #treat punctuation as individual words
        for c in punctuation:
            corpus = [z.replace(c, ' %s '%c) for z in corpus]
        corpus = [z.split() for z in corpus] # 二维矩阵,每一行是一个句子所有token的序列
        return corpus

    x_train = cleanText(x_train)
    x_test = cleanText(x_test)
    unsup_reviews = cleanText(unsup_reviews)

    #Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    #我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
    def labelizeReviews(reviews, label_type):
        labelized = []
        for i,v in enumerate(reviews):
            label = '%s_%s'%(label_type,i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')
    unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

    return x_train,x_test,unsup_reviews,y_train, y_test

##读取向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


##对数据进行训练
def train(x_train,x_test,unsup_reviews,size = 400,epoch_num=10):
    #实例DM和DBOW模型
    model_word2vec = gensim.models.Word2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # 使用所有的数据建立词典
    # x_train:二维矩阵,每一行是一个句子所有token的序列
    model_dm.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))
    model_dbow.build_vocab(np.concatenate((x_train, x_test, unsup_reviews)))

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    all_train_reviews = np.concatenate((x_train, unsup_reviews))
    for epoch in range(epoch_num):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm])
        model_dbow.train(all_train_reviews[perm])

    #训练测试数据集
    x_test = np.array(x_test)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])

    return model_dm,model_dbow

##将训练完成的数据转换为vectors
def get_vectors(model_dm,model_dbow):

    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    #获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs,test_vecs

##使用分类器对文本向量进行分类训练
def Classifier(train_vecs,y_train,test_vecs, y_test):
    #使用sklearn的SGD分类器
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print('Test Accuracy: %.2f'%lr.score(test_vecs, y_test))

    return lr

##绘出ROC曲线，并计算AUC
def ROC_curve(lr,y_test):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    pred_probas = lr.predict_proba(test_vecs)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.show()

##运行模块
if __name__ == "__main__":
    #设置向量维度和训练次数
    size,epoch_num = 400,10
    #获取训练与测试数据及其类别标注
    x_train,x_test,unsup_reviews,y_train, y_test = get_dataset()
    #对数据进行训练，获得模型
    model_dm,model_dbow = train(x_train,x_test,unsup_reviews,size,epoch_num)
    #从模型中抽取文档相应的向量
    train_vecs,test_vecs = get_vectors(model_dm,model_dbow)
    #使用文章所转换的向量进行情感正负分类训练
    lr=Classifier(train_vecs,y_train,test_vecs, y_test)
    #画出ROC曲线
    ROC_curve(lr,y_test)