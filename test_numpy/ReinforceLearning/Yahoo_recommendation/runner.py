# MIT License
#
# Copyright (c) 2016 las.inf.ethz.ch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Evaluation framework for the Bandit setup (Task4, DM2016)"""

import argparse
import io
import imp
import logging
import numpy as np
#import resource
#import signal
import sys
from policy_disjoint import DisjointPolicy
from policy_eplison_greedy import EplisionGreedy
from policy_hybrid import HybridLinUCB
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

"""
日志的意思是从众多article中选择了一个article后的reward
"""
def process_line(logline):
    chosen = int(logline.pop(7)) # chosen article, pop后将会被删除
    reward = int(logline.pop(7)) # 0 or 1
    time = int(logline[0]) # timestamp
    user_features = [float(x) for x in logline[1:7]] # 用户特征
    articles = [int(x) for x in logline[7:]] # list of available article IDs
    return reward, chosen, time, user_features, articles


def evaluate(policy, input_generator):
    score = 0.0
    impressions = 0.0
    n_lines = 0.0

    for line in input_generator:
        n_lines += 1
        reward, chosen, time, user_features, articles = process_line(line.strip().split())
        calculated = policy.recommend(time, user_features, articles)
        # 如果计算的article 与实际上的 article相同
        if calculated == chosen:
            policy.update(reward)
            score += reward
            impressions += 1
        else:
            # 给予反向reward
            policy.update(-1)

    if impressions < 1:
        logger.info("No impressions were made.")
        return 0.0
    else:
        score /= impressions
        logger.info("CTR achieved by the policy [%s]: %.5f impressions:%d" % (policy.__class__.__name__, score, impressions))
        return score


def import_from_file(f):
    """Import code from the specified file"""
    mod = imp.new_module("mod")
    #print(f in mod.__dict__)
    #exec(f in mod.__dict__)
    exec(f)
    return mod


def run(source, log_file, articles_file):
    #policy = import_from_file(source)
    policy_eplision = EplisionGreedy()
    policy_disjoint = DisjointPolicy()
    policy_hybrid = HybridLinUCB()
    articles_np = np.loadtxt(articles_file)
    articles = {} # id -> embedding
    for art in articles_np:
        # id -> embedding
        articles[int(art[0])] = [float(x) for x in art[1:]]
    policy_eplision.set_articles(articles)
    policy_disjoint.set_articles(articles)
    policy_hybrid.set_articles(articles)

    with io.open(log_file, 'rb', buffering=1024*1024*512) as fin:
         evaluate(policy_eplision, fin)
         fin.seek(0)
         evaluate(policy_disjoint, fin)
         fin.seek(0)
         evaluate(policy_hybrid, fin)


"""
令人吃惊的是,epsilon_greedy的表现居然有时很好,不过其对参数epsilon比较敏感

INFO:__main__:CTR achieved by the policy [EplisionGreedy]: 0.04167 impressions:240
INFO:__main__:CTR achieved by the policy [DisjointPolicy]: 0.03310 impressions:4924
INFO:__main__:CTR achieved by the policy [HybridLinUCB]: 0.04195 impressions:4958
"""


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-log_file', nargs='?',const= "data/webscope-logs.txt", default="data/webscope-logs.txt",  help='File containing the log.')
    parser.add_argument(
        '-articles_file', nargs='?', const="data/webscope-articles.txt", default="data/webscope-articles.txt", help='File containing the article features.')
    parser.add_argument(
        '-source_file',  nargs='?',const="policy_disjoint.py",default="policy_disjoint.py", help='.py file implementing the policy.')
    parser.add_argument(
        '-log',  nargs='?',const="log", default="log", help='Enable logging for debugging', action='store_true')
    args = parser.parse_args()
    """
    #args = {}
    np.random.seed(0)
    log_file = "data/webscope-logs.txt"
    articles_file = "data/webscope-articles.txt"
    source_file = "policy_disjoint.py"
    log = "log/1.txt"
    with open(source_file, "r", encoding="utf-8") as fin:
        source = fin.read()
    run(source, log_file, articles_file)