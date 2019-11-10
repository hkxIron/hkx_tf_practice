#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

train, y = make_classification(n_samples=10, n_features=5, n_informative=2, n_redundant=2, n_classes=2, random_state=42)

train = pd.DataFrame(train, columns=['int1', 'int2', 'int3', 's1', 's2'])
train['int1'] = train['int1'].map(int)
train['int2'] = train['int2'].map(int)
train['int3'] = train['int3'].map(int)

train['s1'] = round(np.log(abs(train['s1'] + 1))).map(str)
train['s2'] = round(np.log(abs(train['s2'] + 1))).map(str)
train['clicked'] = y

# libsvm: index从1开始
dummy_train = pd.get_dummies(train)
print("train:\n", train)
print("\n dummy train:\n", dummy_train)
dump_svmlight_file(dummy_train, y, 'svm_output.libsvm', zero_based=False)  # 从1开始编码

#dump_svmlight_file(pd.get_dummies(train[['s1', 's2']]), y, 'svm_output.libsvm', zero_based=False)  # 从1开始编码

"""
train:
    int1  int2  int3    s1    s2  clicked
0     0     0    -1   1.0  -0.0        0
1     1     1     0   0.0   1.0        1
2     0     1    -1  -0.0  -4.0        1
3     1     1     0   1.0   1.0        1
4     1     1     0  -1.0  -2.0        1
5     0    -1     0   1.0  -2.0        0
6    -1    -1     0   1.0  -5.0        0
7     0     0    -1   1.0  -1.0        1
8     0     0     0   1.0  -3.0        0
9    -2    -2     0   1.0   1.0        0

 dummy train:
    int1  int2  int3  clicked  s1_-0.0  s1_-1.0  s1_0.0  s1_1.0  s2_-0.0  \
0     0     0    -1        0        0        0       0       1        1   
1     1     1     0        1        0        0       1       0        0   
2     0     1    -1        1        1        0       0       0        0   
3     1     1     0        1        0        0       0       1        0   
4     1     1     0        1        0        1       0       0        0   
5     0    -1     0        0        0        0       0       1        0   
6    -1    -1     0        0        0        0       0       1        0   
7     0     0    -1        1        0        0       0       1        0   
8     0     0     0        0        0        0       0       1        0   
9    -2    -2     0        0        0        0       0       1        0   

   s2_-1.0  s2_-2.0  s2_-3.0  s2_-4.0  s2_-5.0  s2_1.0  
0        0        0        0        0        0       0  
1        0        0        0        0        0       1  
2        0        0        0        1        0       0  
3        0        0        0        0        0       1  
4        0        1        0        0        0       0  
5        0        1        0        0        0       0  
6        0        0        0        0        1       0  
7        1        0        0        0        0       0  
8        0        0        1        0        0       0  
9        0        0        0        0        0       1  

# 注意: index是从1开始
# 格式: label feat_index:value ...
output file:
0 3:-1 8:1 9:1
1 1:1 2:1 4:1 7:1 15:1
1 2:1 3:-1 4:1 5:1 13:1
1 1:1 2:1 4:1 8:1 15:1
1 1:1 2:1 4:1 6:1 11:1
0 2:-1 8:1 11:1
0 1:-1 2:-1 8:1 14:1
1 3:-1 4:1 8:1 10:1
0 8:1 12:1
0 1:-2 2:-2 8:1 15:1
"""