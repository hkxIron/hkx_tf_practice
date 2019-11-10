#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CountEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, dropna=False, normalize=False):
        """
        :param dropna: 缺失值是否计数，默认计数
        :param normalize: 频数还是频率，默认频数
        """
        self.dropna = dropna
        self.normalize = normalize
        self.map_dict = None

    def fit(self, y):
        self.map_dict = OrderedDict(pd.Series(y)
                                    .value_counts(normalize=self.normalize, dropna=self.dropna)
                                    .to_dict())
        return self

    def transform(self, y):
        """不在训练集的补0，不经常出现补0"""
        return pd.Series(y).map(self.map_dict).fillna(0)


if __name__ == '__main__':
    import numpy as np

    # OrderedDict([(nan, 10), ('a', 3), ('b', 2)])
    # a:出现了3次,所以将所有的a转换成3,nan出现了10次,所以将所有的nan转换成10
    s = ['a', 'a', 'a', 'b', 'b'] + [np.nan] * 10
    #s = ['a', 'a', 'a', 'b', 'b']
    countEncoder = CountEncoder()
    print(countEncoder.fit_transform(s))
    print(countEncoder.map_dict)

    """
    sparse_feats = ["feat1", "feat2", "feat3"]
    # 类别型转换
    for feat in tqdm(sparse_feats): # 注意:需要对每列分别转换
        df[feat] = CountEncoder().fit_transform(df[feat]) # count_encoder
        df['re_'+feat] = CountRankEncoder().fit_transform(df[feat]) # count_encoder + rank
    
    """