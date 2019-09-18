
# coding: utf-8

# In[ ]:


"""
How I made top 0.3% on a Kaggle competition
Getting started with competitive data science can be quite intimidating.
So I wrote this quick overview of how I made top 0.3% on the Advanced Regression 
Techniques competition. If there is interest, I’m happy to do deep dives into the intuition 
behind the feature engineering and models used in this kernel.

I encourage you to fork this kernel, play with the code and enter the competition. Good luck!

If you like this kernel, please give it an upvote. Thank you!


The Goal
1.Each row in the dataset describes the characteristics of a house.
2.Our goal is to predict the SalePrice, given these features.
3.Our models are evaluated on the Root-Mean-Squared-Error (RMSE) between 
the log of the SalePrice predicted by our model, 
and the log of the actual SalePrice. Converting RMSE errors to a log scale ensures that errors in 
predicting expensive houses and cheap houses will affect our score equally.


Key features of the model training process in this kernel:
1.Cross Validation: Using 12-fold cross-validation
2.Models: On each run of cross-validation I fit 7 models (ridge, svr, gradient boosting, random forest, xgboost, lightgbm regressors)
3.Stacking: In addition, I trained a meta StackingCVRegressor optimized using xgboost
4.Blending: All models trained will overfit the training data to varying degrees. Therefore, to make final predictions, I blended their predictions together to get more robust predictions.

Model Performance
1. We can observe from the graph below that the blended model far outperforms the other models, with an RMSLE of 0.075. This is the model I used for making the final predictions.

"""


# In[1]:


from IPython.display import Image
import seaborn as sns


# In[2]:


from mlxtend.regressor import StackingCVRegressor
get_ipython().magic('matplotlib inline')
# 使matplot可以在jupyter中显示 


# In[3]:


# Essentials
import numpy as np
import pandas as pd
import datetime
import random

# Plots
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax



# In[5]:


# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)



# In[6]:


# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
# 设定pandas的选项
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

import os
print(os.listdir("./input"))


# In[87]:


# Read in the dataset as a dataframe
train = pd.read_csv('input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('input/house-prices-advanced-regression-techniques/test.csv')
train.shape, test.shape


# In[8]:


get_ipython().system('ls input/house-prices-advanced-regression-techniques')


# In[9]:


# Preview the data we're working with
train.head()


# In[10]:


train.dtypes


# In[11]:


# 可以看到,pandas加载的列的类型只有int64与object两种类型 


# In[12]:


get_ipython().magic('pinfo sns.distplot')


# In[24]:


# SalePrice: the variable we're trying to predict¶
sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(20, 20))
#Check the new distribution 
sns.distplot(train['SalePrice'], color="b", hist=True, kde=True, rug=True) # kde=True,默认为估计概率密度函数
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)


# In[26]:


# Skew and kurt
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())


# In[28]:


# Let's visualize some of the features in the dataset
# Finding numeric features
# 找出所有的数值类型
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in train.columns:
    if train[i].dtype in numeric_dtypes: # 必须为数值类型 
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass
        else:
            numeric.append(i)       


# In[29]:


train.columns


# In[30]:


[(col, train[col].dtype ) for col in numeric]


# In[33]:


numeric


# In[37]:


train[numeric].head()


# In[38]:


list(train[numeric]) # 是获取DataFrame的所有列


# In[45]:


# visualising some more outliers in the data values
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 150))
plt.subplots_adjust(right=2) # adjust不可省略,否则各个图片之间会重叠在一起
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(train[numeric]), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    # 画出所有的数值feature与SalePrice之间的关系
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)
        
    plt.xlabel('{}'.format(feature), size=15, labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5) # 
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})  # 标题
        
plt.show()


# In[47]:


train.head()


# In[48]:


get_ipython().magic('pinfo train.corr')


# In[58]:


# and plot how the features are correlated to each other, and to SalePrice
corr = train.corr() # 计算 pearson相关系数,只包括数值特征, Compute pairwise correlation of columns, excluding NA/null values
# 不过在图中,并没有看到类别特征 Alley,lotShape, landContour
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Blues", square=True)


# In[62]:


# Let's plot how SalePrice relates to some of the features in the dataset
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
data.head()


# In[63]:


# 对于类别型 的数据,使用 boxplot 
# 画出装修材料与价格这间的关系
data = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(10, 10))
fig = sns.boxplot(x=train['OverallQual'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


# In[57]:


get_ipython().magic('pinfo sns.boxplot')


# In[ ]:


get_ipython().magic('matplotlib inline')


# In[67]:


# YearBuilt,修建时间与价格之间的关系
data = pd.concat([train['SalePrice'], train['YearBuilt']], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=train['YearBuilt'], y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


# In[71]:


# Total square feet of basement area, 对于数值类型 ,画出散点图
data = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
data.plot.scatter(x='TotalBsmtSF', y='SalePrice', alpha=0.5, ylim=(0,800000));


# In[72]:


#  Lot size in square feet
data = pd.concat([train['SalePrice'], train['LotArea']], axis=1)
data.plot.scatter(x='LotArea', y='SalePrice', alpha=0.3, ylim=(0,800000));


# In[73]:


#  Above grade (ground) living area square feet
data = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
data.plot.scatter(x='GrLivArea', y='SalePrice', alpha=0.3, ylim=(0,800000));


# In[77]:


train_ID = train['Id']
train_ID.head()


# In[78]:


type(train_ID)


# In[79]:


# Remove the Ids from train and test, as they are unique for each row and hence not useful for the model
train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
train.shape, test.shape


# In[80]:


"""
Feature Engineering
Let's take a look at the distribution of the SalePrice.
"""


# In[89]:


sns.set_style("white")
sns.set_color_codes(palette='deep')
f, ax = plt.subplots(figsize=(8, 7))
#Check the new distribution 
sns.distplot(train['SalePrice'], color="b");  # 画出分布直方图以及概率密度拟合图
ax.xaxis.grid(False)
ax.set(ylabel="Frequency")
ax.set(xlabel="SalePrice")
ax.set(title="SalePrice distribution")
sns.despine(trim=True, left=True)
plt.show()  # 可以发现,图像有一定的斜度,并非正态分布


# In[90]:


"""
The SalePrice is skewed to the right. This is a problem because most ML models 
don't do well with non-normally distributed data. We can apply a log(1+x) tranform to fix the skew.
"""


# In[93]:


train["SalePrice_bak"] = train["SalePrice"]
train["SalePrice_sqrt"] = np.sqrt(train["SalePrice"])


# In[101]:


# log(1+x) transform
train["SalePrice_log"] = np.log1p(train["SalePrice"])


# In[103]:


def plot_dist_with_sale(data:pd.core.series.Series, title=""):
    # Let's plot the SalePrice again.
    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(8, 7))
    #Check the new distribution 
    sns.distplot(data , fit=norm, color="b");

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)  # 计算最优的均值以及方差
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel="SalePrice")
    ax.set(title=title)
    sns.despine(trim=True, left=True)
    plt.show()


# In[104]:


plot_dist_with_sale(train["SalePrice_bak"] , "origin_sale_price")


# In[106]:


plot_dist_with_sale(train["SalePrice_sqrt"] , "sqrt_sale_price")  # sqrt以及log均可使数据分布成为正态分布


# In[108]:


plot_dist_with_sale(train["SalePrice_log"] , "log_sale_price") # log变换后的正态分布最好,因此使用log变换


# In[112]:


train["SalePrice"] = np.log1p(train["SalePrice_bak"])
# The SalePrice is now normally distributed, excellent!
plot_dist_with_sale(train["SalePrice"] , "transformed_sale_price") 
train_sale_price_bak = np.copy(train["SalePrice_bak"])


# In[116]:


#train.drop(["SalePrice_bak"], inplace=True)
train_sale_price_bak


# In[120]:


# Remove outliers
# 通过以上图的分析,将一些离群点移除,比如: OverallQual - SalePrice图中的离群点
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True) 
# GrLivArea- SalePrice图中右下角的点移除
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True) # 移除部分数据之后,重新计算 新的index


# In[121]:


train['SalePrice']


# In[123]:


train_labels = train['SalePrice'].reset_index(drop=True)
train_labels


# In[124]:


# Split features and labels
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape


# In[125]:


train_features.shape, test_features.shape


# In[134]:


train_features.drop(["SalePrice_bak", "SalePrice_sqrt","SalePrice_log"], inplace=True, axis=1)


# In[135]:


list(train_features)


# In[136]:


list(test_features)


# In[132]:


# Fill missing values


# In[137]:


len(list(train_features))


# In[139]:


type(train_features)


# In[142]:


all_features["PoolQC"].isnull().mean()


# In[217]:


all_features["LotFrontage"].isnull().head()


# In[218]:


all_features.shape


# In[149]:


all_features["LotFrontage"].ix[0],  all_features["LotFrontage"].dtypes


# In[140]:


# determine the threshold for missing values
def percent_missing(df : pd.core.frame.DataFrame):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))  #获取所有的列的list
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]


# In[150]:


# Visualize missing values
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
sns.set_color_codes(palette='deep')
missing = round(train.isnull().mean()*100,2)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(color="b") # 画出柱状图
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Percent of missing values")
ax.set(xlabel="Features")
ax.set(title="Percent missing data by feature")
sns.despine(trim=True, left=True)


# In[151]:


"""
We can now move through each of the features above and impute the missing values for each of them.
"""


# In[153]:


# Some of the non-numeric predictors are stored as numbers; convert them into strings 
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)  # 将 这种类别属性转换成 str类型
all_features['YrSold'] = all_features['YrSold'].astype(str)  # 年份也要转换
all_features['MoSold'] = all_features['MoSold'].astype(str) # 月份


# In[161]:


all_features['Exterior1st'].head()


# In[163]:


all_features['Exterior1st'].mode()[0] # mode,是众数的意思,就是出现最多的那个值


# In[189]:


pd.DataFrame(all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x:x)).head()


# In[188]:


pd.DataFrame(all_features.groupby('MSSubClass')['MSZoning'].transform(lambda x:x.mode()[0])).head()


# In[187]:


two_col_head=all_features.ix[:, ["MSSubClass","MSZoning"]].ix[0:5,:]
two_col_head.ix[0,:]=[60, "RM"]
two_col_head.ix[1,:]=[60, "RM"]
two_col_head.ix[2,:]=[60, "RL"]
two_col_head.ix[3,:]=[70, "RL"]
two_col_head.ix[4,:]=[70, "RL"]
two_col_head.ix[5,:]=[70, "RM"]
two_col_head


# In[190]:


pd.DataFrame(two_col_head.groupby('MSSubClass')['MSZoning'].transform(lambda x:x.mode()[0]))


# In[164]:


# 之所以 要处理nan类型 ,就是因为我们不想丢弃这些样本
def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0]) # 默认值给众数也是不错的选择
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    # 按类别 MSSubClass聚合后, 选取MSZonging的众数,
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0])) 
    
    # the data description stats that NA refers to "No Pool"
    features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:# str 类型 
            objects.append(i)
    features.update(features[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

all_features = handle_missing(all_features)


# In[191]:


# Let's make sure we handled all the missing values
missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]


# In[192]:


# There are no missing values anymore!
# Fix skewed features


# In[193]:


# Fetch all numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)


# In[202]:


# Create box plots for all numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(10, 20))
ax.set_xscale("log")  # 横轴是log 后的
ax = sns.boxplot(data=all_features[numeric] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# In[195]:


# Find skewed numerical features
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)


# In[196]:


#We use the scipy function boxcox1p which computes the Box-Cox transformation. 
#The goal is to find a simple transformation that lets us normalize data.


# In[201]:


get_ipython().magic('pinfo boxcox_normmax')


# In[199]:


get_ipython().magic('pinfo boxcox1p')


# In[197]:


# Normalize skewed features
# 将这些倾斜的数据通过box-cox转换为为近正态变量
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1)) # 计算 1+x 的boxcox变换


# In[203]:


# Let's make sure we handled all the skewed values
sns.set_style("white")
f, ax = plt.subplots(figsize=(10, 20))
ax.set_xscale("log")
ax = sns.boxplot(data=all_features[skew_index] , orient="h", palette="Set1")
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# In[206]:


#  查看变换后的斜度变化
# 将斜度>0.5的特征选出来
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)
# 可以看出,变换后,值的斜度大大减少


# In[210]:


"""
All the features look fairly normally distributed now.

加入人工设计的特征

Create interesting features
ML models have trouble recognizing more complex patterns 
(and we're staying away from neural nets for this competition), 
so let's help our models out by creating a few features based on our intuition about the dataset,
e.g. total area of floors, bathrooms and porch area of each house.

"""


# In[212]:


all_features['ScreenPorch'].head()


# In[211]:


all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')
all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1
all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1
all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1
all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1
#将绝对年份转换为相对年份
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
all_features = all_features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])

# 负数转为 较小的正数, 有些数据为错误 ,需要将其修正,如面积不可能为负数
all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# In[215]:


"""
Feature transformations

Let's create more features by calculating the log and square transformations of our numerical features.
We do this manually, because ML models won't be able to reliably tell if log(feature) or feature^2
is a predictor of the SalePrice.
"""


# In[219]:


type(all_features)


# In[229]:


def logs(res:pd.core.frame.DataFrame, col_names:list):
    m = res.shape[1]
    for col in col_names:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[col])).values)  # 给dataFrame添加人工特征新列
        res.columns.values[m] = col + '_log'
        m += 1 # 此处m有更新
    return res

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

all_features = logs(all_features, log_features)


# In[230]:


all_features.shape


# In[226]:


all_features.columns


# In[241]:


def squares(res:pd.core.frame.DataFrame, ls:list):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'  #添加平方项
        m += 1
    return res 

squared_features = ['YearRemodAdd', 'LotFrontage_log', 
              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
              'GarageCars_log', 'GarageArea_log']
all_features = squares(all_features, squared_features)


# In[242]:


"""
Encode categorical features
Numerically encode categorical features because most models can only handle numerical features.
"""


# In[243]:


# 将类别特征转换成数字变量(数值特征不会转换)
all_features = pd.get_dummies(all_features).reset_index(drop=True)
all_features.shape


# In[236]:


get_ipython().magic('pinfo pd.get_dummies')


# In[244]:


all_features.head()


# In[239]:


all_features.shape


# In[240]:


# Remove any duplicated column names
all_features = all_features.loc[:,~all_features.columns.duplicated()]


# In[245]:


# Recreate training and test sets


# In[248]:


# 特征处理完之后, 分离train/test集
X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape


# In[249]:


# Visualize some of the features we're going to train our models on.


# In[250]:


# Finding numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        if i in ['TotalSF', 'Total_Bathrooms','Total_porch_sf','haspool','hasgarage','hasbsmt','hasfireplace']:
            pass
        else:
            numeric.append(i)     
# visualising some more outliers in the data values
fig, axs = plt.subplots(ncols=2, nrows=0, figsize=(12, 150))
plt.subplots_adjust(right=2)
plt.subplots_adjust(top=2)
sns.color_palette("husl", 8)
for i, feature in enumerate(list(X[numeric]), 1):
    if(feature=='MiscVal'):
        break
    plt.subplot(len(list(numeric)), 3, i)
    sns.scatterplot(x=feature, y='SalePrice', hue='SalePrice', palette='Blues', data=train)
        
    plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    plt.ylabel('SalePrice', size=15, labelpad=12.5)
    
    for j in range(2):
        plt.tick_params(axis='x', labelsize=12)
        plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(loc='best', prop={'size': 10})
        
plt.show()


# In[255]:


print("""
Train a model

Key features of the model training process:
Cross Validation: Using 12-fold cross-validation
Models: On each run of cross-validation I fit 7 models (ridge, svr, gradient boosting, random forest, xgboost, lightgbm regressors)
Stacking: In addition, I trained a meta StackingCVRegressor optimized using xgboost
Blending: All models trained will overfit the training data to varying degrees. Therefore, to make final predictions,
I blended their predictions together to get more robust predictions.

Setup cross validation and define error metrics

""")


# In[256]:


# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True)


# In[258]:


# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
     # k折交叉验证
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf)) 
    return (rmse)


# In[259]:


#Setup models


# In[260]:


# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression',  #线性回归
                       num_leaves=6, # 每颗树6个叶子
                       learning_rate=0.01, 
                       n_estimators=7000, #训练7000颗树
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000, # 6000棵树
                       max_depth=4, # 最大深度为4
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',  # 线性回归
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)


# In[262]:


get_ipython().magic('pinfo RobustScaler')


# In[266]:


# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
# 有较多异常值的情况下, 不适合减均值除标准差,需要用RobustScaler先变换
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf)) #岭回归( 平方正则)

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000, # 6000棵树
                                learning_rate=0.01,
                                max_depth=4,  #  树的最大深度
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200, # 1200棵树
                          max_depth=15, #  树的最大深度
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost, # 第二层的元学习器为 xgboost
                                use_features_in_secondary=True)


# In[273]:


"""
Train models

Get cross validation scores for each model

单独使用lightgbm k 折交叉验证训练, 看看各个模型的预测分数为多少,用于查看各个模型的预测表现
"""


# In[271]:


scores = {}
# 单独使用lightgbm k 折交叉验证训练, 看看各个模型的预测分数为多少,用于查看各个模型的预测表现
score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))  # k折的方差与均值
scores['lgb'] = (score.mean(), score.std())


# In[269]:


score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['xgb'] = (score.mean(), score.std())


# In[278]:


score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['svr'] = (score.mean(), score.std())


# In[279]:


scores


# In[280]:


score = cv_rmse(ridge)
print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['ridge'] = (score.mean(), score.std())


# In[281]:


score = cv_rmse(rf)
print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['rf'] = (score.mean(), score.std())


# In[282]:


score = cv_rmse(gbr)
print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['gbr'] = (score.mean(), score.std())


# In[283]:


"""
使用stack模型拟合
Fit the models
"""


# In[284]:


"""
使用7个模型进行stack,这次是使用所有的数据,而非k折交叉了
"""
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))


# In[ ]:


print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, train_labels)


# In[ ]:


print('xgboost')
xgb_model_full_data = xgboost.fit(X, train_labels)


# In[ ]:


print('Svr')
svr_model_full_data = svr.fit(X, train_labels)


# In[ ]:


print('Ridge')
ridge_model_full_data = ridge.fit(X, train_labels)


# In[ ]:


print('RandomForest')
rf_model_full_data = rf.fit(X, train_labels)


# In[ ]:


print('GradientBoosting')
gbr_model_full_data = gbr.fit(X, train_labels)


# In[ ]:


#Blend models and get predictions


# In[ ]:


# Blend models in order to make the final predictions more robust to overfitting
def blended_predictions(X):
    return ((0.1 * ridge_model_full_data.predict(X)) +             (0.2 * svr_model_full_data.predict(X)) +             (0.1 * gbr_model_full_data.predict(X)) +             (0.1 * xgb_model_full_data.predict(X)) +             (0.1 * lgb_model_full_data.predict(X)) +             (0.05 * rf_model_full_data.predict(X)) +             (0.35 * stack_gen_model.predict(np.array模 #  stack模型给了最高的权重


# In[ ]:


# Get final precitions from the blended model
blended_score = rmsle(train_labels, blended_predictions(X))
scores['blended'] = (blended_score, 0)
print('RMSLE score on train data:')
print(blended_score)


# In[ ]:


# Identify the best performing model


# In[ ]:


# Plot the predictions for each model
sns.set_style("white")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# In[ ]:


""" 
We can observe from the graph above that the blended
model far outperforms the other models, with an RMSLE of 0.075. 

This is the model I'll use for making the final predictions. 
"""


# In[ ]:


# Submit predictions


# In[275]:


# Read in sample_submission dataframe
submission = pd.read_csv("input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.shape


# In[276]:


submission.head()


# In[274]:


# Append predictions from blended models
submission.iloc[:,1] = np.floor(np.expm1(blended_predictions(X_test))) # 使用exp(x) -1 将我们转换后的价格真实数值恢复


# In[ ]:


# Fix outleir predictions 
q1 = submission['SalePrice'].quantile(0.0045) #  0.0045分位点
q2 = submission['SalePrice'].quantile(0.99)  # 99%分位点
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission_regression1.csv", index=False)


# In[277]:


# Scale predictions
submission['SalePrice'] *= 1.001619  # 将最终预测的值进行了缩放
submission.to_csv("submission_regression2.csv", index=False)

