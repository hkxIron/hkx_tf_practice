# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class OutlierEncoder(object):
    """判定和处理异常值方法

    若数据服从正态分布，在3σ原则下，异常值被定义为一组测定值中与平均值的偏差超过3倍标准差的值，

    因为在正态分布的假设下，距离平均值3σ之外的值出现的概率小于0.003
    1.删除异常值记录

    2.将异常值视为缺失值，由缺失值处理方法来处理

    3.用中位数来处理
    4.不处理
    """

    def __init__(self, mode='isf'):
        """
        :param mode: 'isf', 'box_plot', 'zscore_median'
        """
        self.mode = mode

    def transform(self, x: pd.Series, return_bool=True):
        """返回异常值的索引或者0/1"""
        if return_bool:
            return self.__getattribute__(self.mode)(x)
        else:
            return self.__getattribute__(self.mode)(x).astype(int)

    def get_outlier_idx(self, x: pd.Series):
        """投票"""
        _ = [self.isolation_forest(x), self.box_plot(x), self.zscore_median(x)]
        _ = pd.DataFrame(_).sum() > 1
        return np.where(_)[0].tolist()

    def isolation_forest(self, x):
        isolation = IsolationForest(contamination='auto', n_jobs=-1)
        isolation.fit(x.to_frame())
        return isolation.predict(x.to_frame()) == -1

    def box_plot(self, x):
        bound_func = lambda a, b: 2.5 * a - 1.5 * b
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        lower_bound, upper_bound = bound_func(q1, q3), bound_func(q3, q1)
        return ~x.between(lower_bound, upper_bound)

    def zscore_median(self, x, threshold=3):
        """3 sigma"""
        x_median = (x - x.median()).abs()
        MAD = x_median.median()
        zscore = 0.6475 * x_median / MAD
        return zscore > threshold

    # def zscore_mean(self, x, threshold=3):
    #     zscore = (x - x.mean()) / x.std()
    #     return zscore.abs() > threshold


if __name__ == '__main__':
    s = pd.Series([-10, 1, 2, 3, 4, 5, 100])
    print(s)

    outlier = OutlierEncoder('zscore_median')
    print(outlier.get_outlier_idx(s))
    print(outlier.transform(s))
