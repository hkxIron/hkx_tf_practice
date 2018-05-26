#
# This script trains tries to tweak hyperparameters to improve P/R AUC
#

import time
start_time = time.time()

import nltk

import numpy as np

from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.cross_validation import ShuffleSplit

from utils import plot_pr
from utils import load_sanders_data
from utils import tweak_labels
from utils import log_false_positives

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator

from sklearn.naive_bayes import MultinomialNB

from utils import load_sent_word_net

sent_word_net = load_sent_word_net()


import json

poscache_filename = "poscache.json"
try:
    poscache = json.load(open(poscache_filename, "r"))
except IOError:
    poscache = {}

class StructCounter(BaseEstimator):
    def get_feature_names(self):
        return np.array(['sent_neut', 'sent_pos', 'sent_neg',
         'nouns', 'adjectives', 'verbs', 'adverbs',
         'allcaps', 'exclamation', 'question', 'hashtag', 'mentioning'])

    def fit(self, documents, y=None):
        return self

    def _get_sentiments(self, d):
        # http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        #import pdb;pdb.set_trace()
        sent = tuple(d.split())
        if poscache is not None:
            if d in poscache:
                tagged = poscache[d]
            else:
                poscache[d] = tagged = nltk.pos_tag(sent)
        else:
            tagged = nltk.pos_tag(sent)

        pos_vals = []
        neg_vals = []

        nouns = 0.
        adjectives = 0.
        verbs = 0.
        adverbs = 0.

        for w,t in tagged:
            p, n = 0,0
            sent_pos_type = None
            if t.startswith("NN"):
                sent_pos_type = "n"
                nouns += 1
            elif t.startswith("JJ"):
                sent_pos_type = "a"
                adjectives += 1
            elif t.startswith("VB"):
                sent_pos_type = "v"
                verbs += 1
            elif t.startswith("RB"):
                sent_pos_type = "r"
                adverbs += 1

            if sent_pos_type is not None:
                sent_word = "%s/%s"%(sent_pos_type, w)

                if sent_word in sent_word_net:
                    p,n = sent_word_net[sent_word]

            pos_vals.append(p)
            neg_vals.append(n)

        l = len(sent)
        avg_pos_val = np.mean(pos_vals)
        avg_neg_val = np.mean(neg_vals)
        return [1-avg_pos_val-avg_neg_val, avg_pos_val, avg_neg_val,
                nouns/l, adjectives/l, verbs/l, adverbs/l]


    def transform(self, documents):
        obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs = np.array([self._get_sentiments(d) for d in documents]).T

        allcaps = []
        exclamation = []
        question = []
        hashtag = []
        mentioning = []

        for d in documents:
            #import pdb;pdb.set_trace()

            allcaps.append(np.sum([t.isupper() for t in d.split() if len(t)>2]))

            exclamation.append(d.count("!"))
            question.append(d.count("?"))
            hashtag.append(d.count("#"))
            mentioning.append(d.count("@"))

        result = np.array([obj_val, pos_val, neg_val, nouns, adjectives, verbs, adverbs, allcaps,
            exclamation, question, hashtag, mentioning]).T

        return result


def create_union_model(params=None):
    def preprocessor(tweet):
        global emoticons_replaced
        #return tweet.lower()
        repl = {
            # positive emoticons
            "&lt;3": " good ",
            ":D": " good ",
            "8)": " good ",
            ":-)": " good ",
            ":)": " good ",
            ";)": " good ",
            ";-)": " good ",

            # negative emoticons:
            ":/": " bad ",
            ":&gt;": " sad ",
            ":-(": " bad ",
            ":(": " bad ",
            ":S": " bad ",
            ":-S": " bad ",
        }

        for a,b in repl.iteritems():
            tweet = tweet.replace(a,b)

        return tweet.lower()

    tfidf_ngrams = TfidfVectorizer(preprocessor=preprocessor,
                                   analyzer="word")
    struct_stats = StructCounter()
    all_features = FeatureUnion([('struct', struct_stats), ('tfidf', tfidf_ngrams)])
    #all_features = FeatureUnion([('tfidf', tfidf_ngrams)])
    #all_features = FeatureUnion([('struct', struct_stats)])
    clf = MultinomialNB()
    pipeline = Pipeline([('all', all_features), ('clf', clf)])

    if params:
        pipeline.set_params(**params)

    return pipeline


def grid_search_model(clf_factory, X, Y):
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)

    param_grid = dict(all__tfidf__ngram_range=[(1, 1), (1, 2), (1, 3)],
                      all__tfidf__min_df=[1, 2],
                      all__tfidf__stop_words=[None, "english"],
                      all__tfidf__smooth_idf=[False, True],
                      all__tfidf__use_idf=[False, True],
                      all__tfidf__sublinear_tf=[False, True],
                      all__tfidf__binary=[False, True],
                      clf__alpha=[0, 0.01, 0.05, 0.1, 0.5, 1],
                      )

    grid_search = GridSearchCV(clf_factory(),
                               param_grid=param_grid,
                               cv=cv,
                               score_func=f1_score,
                               verbose=10)
    grid_search.fit(X, Y)
    clf = grid_search.best_estimator_
    print clf

    return clf


def train_model(clf, X, Y, name="NB ngram", plot=False):
    # create it again for plotting
    cv = ShuffleSplit(
        n=len(X), n_iter=10, test_size=0.3, indices=True, random_state=0)

    train_errors = []
    test_errors = []

    scores = []
    pr_scores = []
    precisions, recalls, thresholds = [], [], []

    clfs = [] # just to later get the median

    for train, test in cv:
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]

        clf.fit(X_train, y_train)
        clfs.append(clf)

        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        scores.append(test_score)
        proba = clf.predict_proba(X_test)

        fpr, tpr, roc_thresholds = roc_curve(y_test, proba[:, 1])
        precision, recall, pr_thresholds = precision_recall_curve(
            y_test, proba[:, 1])

        pr_scores.append(auc(recall, precision))
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(pr_thresholds)

    if plot:
        scores_to_sort = pr_scores
        median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]

        plot_pr(pr_scores[median], name, precisions[median],
                recalls[median], label=name)

        log_false_positives(clfs[median], X_test, y_test, name)

    summary = (np.mean(scores), np.std(scores),
               np.mean(pr_scores), np.std(pr_scores))
    print "%.3f\t%.3f\t%.3f\t%.3f\t" % summary

    return np.mean(train_errors), np.mean(test_errors)


def print_incorrect(clf, X, Y):
    Y_hat = clf.predict(X)
    wrong_idx = Y_hat != Y
    X_wrong = X[wrong_idx]
    Y_wrong = Y[wrong_idx]
    Y_hat_wrong = Y_hat[wrong_idx]
    for idx in xrange(len(X_wrong)):
        print "clf.predict('%s')=%i instead of %i" %\
            (X_wrong[idx], Y_hat_wrong[idx], Y_wrong[idx])

def get_best_model():
    best_params = dict(all__tfidf__ngram_range=(1, 2),
                       all__tfidf__min_df=1,
                       all__tfidf__stop_words=None,
                       all__tfidf__smooth_idf=False,
                       all__tfidf__use_idf=False,
                       all__tfidf__sublinear_tf=True,
                       all__tfidf__binary=False,
                       clf__alpha=0.01,
                       )

    best_clf = create_union_model(best_params)

    return best_clf

if __name__ == "__main__":
    X_orig, Y_orig = load_sanders_data()
    #from sklearn.utils import shuffle
    #print "shuffle, sample"
    #X_orig, Y_orig = shuffle(X_orig, Y_orig)
    #X_orig = X_orig[:100,]
    #Y_orig = Y_orig[:100,]
    classes = np.unique(Y_orig)
    for c in classes:
        print "#%s: %i" % (c, sum(Y_orig == c))

    print "== Pos vs. neg =="
    pos_neg = np.logical_or(Y_orig == "positive", Y_orig == "negative")
    X = X_orig[pos_neg]
    Y = Y_orig[pos_neg]
    Y = tweak_labels(Y, ["positive"])
    grid_search_model(create_union_model, X, Y)

    print "== Pos/neg vs. irrelevant/neutral =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive", "negative"])

    # best_clf = grid_search_model(create_union_model, X, Y, name="sent vs
    # rest", plot=True)
    grid_search_model(create_union_model, X, Y)

    print "== Pos vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["positive"])
    grid_search_model(create_union_model, X, Y)

    print "== Neg vs. rest =="
    X = X_orig
    Y = tweak_labels(Y_orig, ["negative"])
    grid_search_model(create_union_model, X, Y)

    print "time spent:", time.time() - start_time

    json.dump(poscache, open(poscache_filename, "w"))
