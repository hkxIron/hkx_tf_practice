# blog: http://blog.csdn.net/Walker_Hao/article/details/78995591
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

import numpy as np

from sklearn.linear_model import LogisticRegression

import logging
import sys
import random

# code from the tutorial of the python model logging.
# create a logger, the same name corresponding to the same logger.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create console handler and set level to info
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

# create formatter and add formatter to ch
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

logger.addHandler(ch)   # add ch to the logger


## the code for the doc2vec
class TaggedLineSentence(object):
    """
    sources: [file1 name: tag1 name, file2 name: tag2 name ...]
    privade two functions:
        to_array: transfer each line to a object of TaggedDocument and then add to a list
        perm: permutations
    """
    def __init__(self, sources):
        self.sources = sources

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    # TaggedDocument([word1, word2 ...], [tagx])
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(),
                                       [prefix + '_%s' % item_no]))
        return self.sentences

    def perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)    # Note that this line does not return anything.
        return shuffled


sources = {'test-neg.txt': 'TEST_NEG', 'test-pos.txt': 'TEST_POS',
           'train-neg.txt': 'TRAIN_NEG','train-pos.txt': 'TRAIN_POS',
           'train-unsup.txt': 'TRAIN_UNS'}
sentences = TaggedLineSentence(sources)

# set the parameter and get a model.
# by default dm=1, PV-DM is used. Otherwise, PV-DBOW is employed.
model = Doc2Vec(min_count=1, window=10, size=100,
                sample=1e-4, negative=5, dm=1, workers=7)
model.build_vocab(sentences.to_array())

# train the model
for epoch in range(20):
    logger.info('epoch %d' % epoch)
    model.train(sentences.perm(),
                total_examples=model.corpus_count,
                epochs=model.iter
                )

logger.info('model saved')
model.save('./imdb.d2v')

# load and test the model
logger.info('model loaded')
model = Doc2Vec.load('./imdb.d2v')

logger.info('Sentiment Analysis...')

logger.info('transfer the train document to the vector')
train_arrays = np.zeros((25000, 100))
train_labels = np.zeros(25000)

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    # note that the attribute is model.docvecs
    train_arrays[i], train_arrays[12500+i] = \
        model.docvecs[prefix_train_pos], model.docvecs[prefix_train_neg]
    train_labels[i], train_labels[12500+i] = 1, 0


logger.info('transfer the test document to the vector')
test_arrays = np.zeros((25000, 100))
test_labels = np.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    test_arrays[i], test_arrays[12500 + i] = \
        model.docvecs[prefix_test_pos], model.docvecs[prefix_test_neg]
    test_labels[i], test_labels[12500 + i] = 1, 0

logger.info('Fitting')
classifier = LogisticRegression()
classifier.fit(train_arrays, train_labels)

print("accuracy: "+ str(classifier.score(test_arrays, test_labels)))