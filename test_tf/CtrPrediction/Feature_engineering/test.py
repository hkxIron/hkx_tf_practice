# coding:utf-8
import pandas as pd
import numpy as np

dataPath="../data/"
ids = pd.read_csv(dataPath+'avazu_ctr_test_1000.csv')['id'].values

click = pd.read_csv(dataPath+'/FM_FTRL_v1.csv')['click'].values

print(len(ids))
print(len(click))

pd.DataFrame(np.array([ids, click]).T, columns=['id','click']).to_csv('FM_FTRL_v1.csv', index=False)



