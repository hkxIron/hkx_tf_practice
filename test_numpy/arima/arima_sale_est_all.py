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
import time
#import logging 
#logging.getLogger("requests").setLevel(logging.WARNING)

bPlot=False

def get_pred(fileName,outFile):
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
    series_data=dta_full[['pay_30','pay_29','pay_28','pay_27','pay_26','pay_25','pay_24','pay_23','pay_22','pay_21','pay_20','pay_19','pay_18','pay_17','pay_16','pay_15','pay_14','pay_13','pay_12','pay_11','pay_10','pay_9','pay_8','pay_7','pay_6','pay_5','pay_4','pay_3','pay_2','pay_1']]
    print("series_data shape:"+str(series_data.shape))
    #一阶差分
    series_data=series_data.T
    diff1 = series_data.diff(periods=1,axis=0)
    diff1[diff1.isnull()] =0 
    test_index=[0,1,2,3,4,5]
    diff_test=diff1[test_index]
    if bPlot:
        fig = plt.figure(figsize=(12,8))
        ax1= fig.add_subplot(111)
        ax1.set_title(u"first order diff")
        diff_test.plot(ax=ax1)
    #plt.show()  #要加plt.show

    #二阶差分
    diff2 = diff1[test_index].diff(periods=1,axis=0)
    diff2[diff2.isnull()] =0
    diff2[diff2.isnull()] =0
    if bPlot:
        fig = plt.figure(figsize=(12,8))
        ax2= fig.add_subplot(111)
        ax2.set_title(u"second order diff")
        diff2.plot(ax=ax2)
    #plt.show()  #要加plt.show

    #合适的p,q
    # dta= dta.diff(1)#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
    # dta[0]=0
    N=diff1.shape[1] #每列是一个样本的预测信息
    labelArr=label.as_matrix()
    predArr=np.zeros(N)
    failCnt=0
    for ind in xrange(N):
        dta=diff1[[ind]] #获取第ind列
        predArr[ind],bSuccess=get_each_pred(dta)
        failCnt+=not bSuccess
        error=np.abs(predArr[ind]-label[ind])
        if ind%100==0:
            print("ind:%d pred:%f,label:%f,error:%f"%(ind,predArr[ind],label[ind],error))
    error=np.abs(labelArr-predArr)
    rmse=np.sqrt(np.sum(error**2)/N)
    mae=np.sum(error)/(np.sum(labelArr)+1e-5)
    print("N:%d failCnt:%d failRate:%.4f abs_error:%f rmse:%f mae:%f"%(N,failCnt,failCnt/(N+1.0),np.mean(error),rmse,mae))
    print("save pred to file:",outFile)
    
    np.savetxt(outFile,predArr,fmt='%.4f', delimiter='\t')
    item_id_arr=dta_full[['item_id','stat_ds']].as_matrix().reshape((N,2));
    labelPredError=np.concatenate((item_id_arr,
                                   np.reshape(labelArr,(N,1)),
                                   np.reshape(predArr,(N,1)),
                                   np.reshape(error,(N,1))),
                                   axis=1)
    outFileFull=outFile+"_full"
    print("save all out data to file:",outFileFull)
    np.savetxt(outFileFull,labelPredError,fmt='%.2f', delimiter='\t')

def get_each_pred(dta):
    bSuccess=True
    if bPlot:
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

#     arma_mod20 = sm.tsa.ARMA(dta,(7,2)).fit()
#     print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)

    #fit
    try:    
        arma_mod20 = sm.tsa.ARMA(dta,(7,0)).fit()
        bSuccess=True
        #print("arma_mod20:",arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
    except Exception,e:
        print("arma fit error!")
        bSuccess=False
        return 0,bSuccess
    """
    arma_mod10 = sm.tsa.ARMA(dta,(7,0)).fit()
    arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
    arma_mod40 = sm.tsa.ARMA(dta,(7,1)).fit()
    arma_mod50 = sm.tsa.ARMA(dta,(8,0)).fit()
    print("arma_mod10:",arma_mod10.aic,arma_mod10.bic,arma_mod10.hqic)
    print("arma_mod30:",arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
    print("arma_mod40:",arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
    print("arma_mod50:",arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)
    """

    #我们对ARMA(7,0)模型所产生的残差做自相关图
    if bPlot:
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        ax1.set_title(u"arma_mod20 acf")
        fig = sm.graphics.tsa.plot_acf(arma_mod20.resid.values.squeeze(),  ax=ax1)
        ax2 = fig.add_subplot(212)
        ax2.set_title(u"arma_mod20 pacf")
        fig = sm.graphics.tsa.plot_pacf(arma_mod20.resid, ax=ax2)

    #德宾-沃森（Durbin-Watson）检验
    #print("Durbin-Watson value:",sm.stats.durbin_watson(arma_mod20.resid.values)) # :2.0760485828, 而接近于２时，则不存在（一阶）自相关性

    #观察是否符合正态分布
    resid = arma_mod20.resid#残差
    if bPlot:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        fig = qqplot(resid, line='q', ax=ax, fit=True)
        plt.show()  #要加plt.show

    """
    #Ljung-Box检验
    r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1,41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))
    """

    predict_sunspots = arma_mod20.predict('20170508','20170509',dynamic=True)
    print(predict_sunspots)
    if bPlot:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax = dta.ix[dta.index[0]:].plot(ax=ax)
        ax.set_title(u"predict 1st diff")
        predict_sunspots.plot(ax=ax)
        #plt.show()

    last=dta.irow(dta.shape[0]-1)
    pred_value=last.values[0]+predict_sunspots[0]
    return pred_value,bSuccess

if __name__ == "__main__":
    print("start time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    fileName="data/item_pay_seq_stat_sample_1m_info.txt"
    outPred="data/item_pay_seq_stat_sample_1m_info_pred.txt"
    get_pred(fileName,outPred)
    print("end time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


