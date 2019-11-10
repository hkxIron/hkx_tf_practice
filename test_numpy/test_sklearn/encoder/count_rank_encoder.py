#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CountRankEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, topn=None):
        """
        :param topn: 仅保留topn个类别
        """
        self.topn = topn
        self.map_dict = None

    """
    orgin count map: NaN    0.545455
    b      0.181818
    a      0.181818
    c      0.090909
    dtype: float64
    Coverage: 72.73%
    
    只保留那些top的特征
    """
    def fit(self, y):
        count_map = pd.Series(y).value_counts(True, dropna=False)  # 计数编码
        print("orgin count map:", count_map)
        if self.topn:
            count_map = count_map[:self.topn]
            print(f"Coverage: {count_map.sum() * 100:.2f}%")

        self.map_dict = OrderedDict(count_map.rank(method='first').to_dict())  # rank 合理？
        return self

    def transform(self, y):
        """不在训练集的补0，不经常出现补0"""
        return pd.Series(y).map(self.map_dict).fillna(0)


if __name__ == '__main__':
    import numpy as np

    s = ['a', 'a', 'b', 'b', 'c'] + [np.nan] * 6
    counter_rank_encoder = CountRankEncoder(2)

    print("transformed data:\n", counter_rank_encoder.fit_transform(s))
    print("map_dict:", counter_rank_encoder.map_dict)