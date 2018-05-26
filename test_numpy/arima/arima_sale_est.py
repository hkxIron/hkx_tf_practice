#coding:utf-8
"""
This is a batched LSTM forward and backward pass
"""
from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import logging 
# dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
# 6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
# 10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
# 12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
# 13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
# 9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
# 11999,9390,13481,14795,15845,15271,14686,11054,10395
# ]

logging.getLogger("requests").setLevel(logging.WARNING)

fileName="E:\\kp\\lunaWorkspace\\mlPractice\\data\\item_pay_seq_stat_sample_1m_info.txt"
#np_data=np.loadtxt(fileName, dtype=np.int64,delimiter='\t')
dta_full = pd.read_csv(fileName,encoding='utf8',sep='\t')
print("data shape:"+str(dta_full.shape))
print("data columns:"+str(dta_full.columns))
label=dta_full.pay
#np_matrix=dta_full
#dta=np_data[:,5:35]
#print("data shape:"+str(dta.shape))
#dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2090'))
#dta.plot(figsize=(12,8))
#plt.show()  #要加plt.show
#series_data=dta_full.ix(['pay_30','pay_29','pay_28','pay_27','pay_26','pay_25','pay_24','pay_23','pay_22','pay_21','pay_20','pay_19','pay_18','pay_17','pay_16','pay_15','pay_14','pay_13','pay_12','pay_11','pay_10','pay_9','pay_8','pay_7','pay_6','pay_5','pay_4','pay_3','pay_2','pay_1'])
print("start!")
series_data=dta_full[['pay_30','pay_29','pay_28','pay_27','pay_26','pay_25','pay_24','pay_23','pay_22','pay_21','pay_20','pay_19','pay_18','pay_17','pay_16','pay_15','pay_14','pay_13','pay_12','pay_11','pay_10','pay_9','pay_8','pay_7','pay_6','pay_5','pay_4','pay_3','pay_2','pay_1']]
print("series_data shape:"+str(series_data.shape))
#一阶差分
series_data=series_data.T
fig = plt.figure(figsize=(12,8))
ax1= fig.add_subplot(111)
ax1.set_title(u"first order diff")
diff1 = series_data.diff(periods=1,axis=0)
diff1[diff1.isnull()] =0 
diff_test=diff1[[0,1,2,3,4,5]]
diff_test.plot(ax=ax1)
#plt.show()  #要加plt.show

#二阶差分
fig = plt.figure(figsize=(12,8))
ax2= fig.add_subplot(111)
ax2.set_title(u"second order diff")
diff2 = diff1[[0,1,2,3,4,5]].diff(periods=1,axis=0)
diff2[diff2.isnull()] =0
diff2[diff2.isnull()] =0
diff2.plot(ax=ax2)
#plt.show()  #要加plt.show


#合适的p,q
# dta= dta.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
# dta[0]=0
dta=diff_test[[0]]
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
ax1.set_title("acf")
fig = sm.graphics.tsa.plot_acf(dta,ax=ax1)
ax2 = fig.add_subplot(212)
ax1.set_title("pacf")
fig = sm.graphics.tsa.plot_pacf(dta,ax=ax2)
#plt.show()

#------------------
#dta.index=pd.Index(sm.tsa.datetools.dates_from_range('2001',length=30))
dta.index=pd.Index(pd.date_range(end='20170507',periods=30))

arma_mod20 = sm.tsa.ARMA(dta,(7,2)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
"""
arma_mod20 = sm.tsa.ARMA(dta,(7,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
arma_mod40 = sm.tsa.ARMA(dta,(7,1)).fit()
print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
arma_mod50 = sm.tsa.ARMA(dta,(8,0)).fit()
print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)
"""

#我们对ARMA(7,0)模型所产生的残差做自相关图
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
ax1.set_title(u"arma_mod20 acf")
fig = sm.graphics.tsa.plot_acf(arma_mod20.resid.values.squeeze(),  ax=ax1)
ax2 = fig.add_subplot(212)
ax2.set_title(u"arma_mod20 pacf")
fig = sm.graphics.tsa.plot_pacf(arma_mod20.resid, ax=ax2)

#德宾-沃森（Durbin-Watson）检验
print(sm.stats.durbin_watson(arma_mod20.resid.values)) # :2.0760485828, 而接近于２时，则不存在（一阶）自相关性

#观察是否符合正态分布
resid = arma_mod20.resid#残差
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

"""
#Ljung-Box检验
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
"""

predict_sunspots = arma_mod20.predict('20170508','20170510',dynamic=True)
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.ix[dta.index[0]:].plot(ax=ax)
ax.set_title(u"predict 1st diff")
predict_sunspots.plot(ax=ax)


last=dta.irow(dta.shape[0]-1)
pred_value=last[0]+predict_sunspots[0]
print("final predict:%f"%(pred_value));
error=np.abs(pred_value-label[0])
print("pred:%f,label:%f,error:%f"%(pred_value,label[0],error))
plt.show()  #要加plt.show


