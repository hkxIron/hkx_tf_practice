#coding:utf-8
"""
This is a batched LSTM forward and backward pass
"""
#-*- coding: utf-8 -*-
#arima时序模型
import pandas as pd

#参数初始化
discfile = 'E:\\kp\\lunaWorkspace\\mlPractice\\test.csv'
forecastnum = 5

#读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式
#data = pd.read_excel(discfile, index_col = u'日期')
ori_data = pd.read_csv(discfile,encoding='utf8')
#data = pd.read_csv(discfile,encoding='utf8') #['sale']

#时序图
import matplotlib.pyplot as plt

#用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei'] 
#用来正常显示负号

plt.rcParams['axes.unicode_minus'] = False 
ori_data.plot()
plt.show()

data=ori_data['sale']
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data)
#plot_acf(data).show()
plt.show()

#平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print('ADF result:', ADF(ori_data['sale']))

#返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

#差分后的结果
D_data = data.diff().dropna()
D_data.columns = ['sale_diff']
#时序图
D_data.plot() 
plt.show()

#-------------
#自相关图
plot_acf(D_data).show()
plt.show()
from statsmodels.graphics.tsaplots import plot_pacf
#偏自相关图
plot_pacf(D_data).show()
#平稳性检测

print(u'差分序列的ADF检验结果为：', ADF(D_data['sale_diff'])) 

#Pdf值小于两个水平值，p值显著小于0.05，一阶差分后序列为平稳序列。

 
#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox

#返回统计量和p值
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1)) 
from statsmodels.tsa.arima_model import ARIMA
#定阶

#一般阶数不超过length/10

pmax = int(len(D_data)/10) 

#一般阶数不超过length/10
qmax = int(len(D_data)/10) 

bic_matrix = [] 
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
       #存在部分报错，所以用try来跳过报错。
        try: 
            tmp.append(ARIMA(data, (p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

#从中可以找出最小值
bic_matrix = pd.DataFrame(bic_matrix) 
#先用stack展平，然后用idxmin找出最小值位置。
p,q = bic_matrix.stack().idxmin() 
print(u'BIC最小的p值和q值为：%s、%s' %(p,q))

 #建立ARIMA(0, 1, 1)模型
model = ARIMA(data, (p,1,q)).fit() 
#给出一份模型报告
model.summary2() 

#作为期5天的预测，返回预测结果、标准误差、置信区间。
model.forecast(5)




