# padans数据处理工具函数
import pandas as pd
import numpy as np
from IPython.display import Image
import seaborn as sns
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import datetime
import random
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import os
from time import strftime, localtime
EPS=1e-6

# 带时间打印的 print
def printf(*args, sep=' ', end='\n'): # known special case of print
    print("["+get_time_str()+"]", end=' ')
    print(*args, sep=sep, end=end)

def get_time_str():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def save_data_to_csv(df:pd.DataFrame, out_file, index=False):
    df.to_csv(out_file, encoding="utf-8", index=False, header=True)
    print("save data to file:{} shape:{}".format(out_file, df.shape))

def show_label_info(data, label_name="label", name="input_data"):
    print("{} size:{}".format(name, data.shape))
    if data is pd.Series or len(data.shape)==1 or data.shape[1] == 1:
        print("label info:", data.iloc[:,0].value_counts())
    else:
        print("label info:",data[label_name].value_counts())

def load_data(file_name, sample_num=0):
    print("read file: {} ...".format(file_name))
    if sample_num>0:
        data = pd.read_csv(file_name, nrows=sample_num, encoding="utf-8")
    else:
        data = pd.read_csv(file_name, encoding="utf-8")
    print("file:{} input data size:{}".format(file_name, data.shape))
    return data

# pd.set_option('max_rows', 1000)
def plot_float_data_distribute(x_input:pd.Series, input_title:str):
    # 去除里面的na
    x= x_input.dropna(inplace=False)

    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(20, 20))
    #Check the new distribution
    try:
        sns.distplot(x.dropna(), color="b", hist=True, kde=True, rug=True ,fit=norm) # kde=True,默认为估计概率密度函数
        # Get the fitted parameters used by the function
        (mu, sigma) = norm.fit(x)  # 计算最优的均值以及方差
        print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

        #Now plot the distribution
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
        ax.xaxis.grid(False)
        ax.set(ylabel="Frequency")
        ax.set(xlabel=input_title)
        ax.set(title=input_title+" distribution")
        sns.despine(trim=True, left=True)
    except Exception as ex:
        print("plot dist error for:", x_input.name, ex)

def append_str_file(file_name, suffix):
    sep="."
    arr = file_name.split(sep)
    return arr[0]+"_"+suffix+sep+arr[1]

def fillna_all_by_str(data:pd.DataFrame, value="unk"):
    strs = []
    for i in data.columns:
        #print(data[i].dtype)
        if data[i].dtype == object:
            strs.append(i)
    print("fill cols:"+str(strs)+"\nvalues:"+value)
    data.update(data[strs].fillna(value)) # 回写

def convert_continus_to_discrete(series:pd.Series, prefix="", suffix_list=range(10), category_to_str=False):
    #assert series.dtype != object
    label_list = [prefix + str(x) for x in suffix_list]
    cut_series, bins = pd.cut(series, bins=len(label_list), retbins=True, labels=label_list, precision=3)
    print("value count:", cut_series.value_counts())
    print("size:",len(bins),"cut bins:", bins)
    if category_to_str:
        cut_series = cut_series.astype(str).apply(lambda x:x.replace("nan", "unk").replace("NaN", "unk"))
    return cut_series, bins, pd.value_counts(cut_series)

def get_qcut_by_value_count(ser:pd.Series, prefix="", suffix_list=range(10)):
    value_cnt = ser.value_counts()
    print("origin value cnt:", value_cnt)
    label_list = [prefix + str(x) for x in suffix_list]
    cut_series, bins = pd.qcut(value_cnt, q=len(label_list), labels=label_list, retbins=True)
    print("qcut value cnt:", cut_series.value_counts()) # 或者:cut_series.groupby(cur_series).count()
    return cut_series, bins

def fillna_each_by_str(data:pd.DataFrame, cols,value="unk"):
    for x in cols:
        print("fill col:"+x+" by "+value)
        data[x].fillna(value, inplace=True)

def get_all_numberic_columns(df:pd.DataFrame, exclude_columns=[]):
    # 找出所有的数值类型
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:  # 必须为数值类型
            if i in exclude_columns:
                pass
            else:
                numeric.append(i)
    return numeric

def get_str_values(ser:pd.Series):
    assert ser.dtype == object # 必须是str
    value_cnt = ser.value_counts()
    print("origin value cnt:\n", value_cnt)
    sorted_value_cnt = sorted(dict(value_cnt).items(), key =lambda x:x[1], reverse=True) # feat11: A -> 10, B -> 9, C -> 7
    values = [x[0] for x in sorted_value_cnt] # A,B,C,D
    return values

def get_all_str_columns(df:pd.DataFrame):
    expect_types = ['object']
    return_cols = []
    for i in df.columns:
        if df[i].dtype in expect_types:  # 必须为数值类型
            return_cols.append(i)
    return return_cols

def show_all_cols_info(df:pd.DataFrame):
    col_num = len(df.columns)
    miss_percent = get_missing_percent(df)
    print("cols num:{} details:{}".format(col_num, "\n".join([str(x) for x in miss_percent])))

# determine the threshold for missing values
def get_missing_percent(df:pd.DataFrame):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data).columns)
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: [round(data[df_cols[i]].isnull().mean() * 100, 2), data[df_cols[i]].dtype]}) # update:将一个dict值全放入另一个dict
    return dict_x

def get_and_print_missing_precent(df:pd.DataFrame):
    print("shape:", df.shape)
    missing = get_missing_percent(df)
    cols_miss = sorted(missing.items(), key=lambda x: x[1][0], reverse=True)
    print('Percent of missing data:')
    print("\n".join([str(x) for x in cols_miss]))
    return cols_miss

def to_str_reserve_null(x):
    if not pd.isnull(x):
        return str(x)
    else:
        return x

def lower_str_value(x, case_sensitive=False):
    if not pd.isnull(x) and not case_sensitive:
        return x.lower()
    else:
        return x

"""
如果key的count太小,那么key被转化为 "other"

如果key为isnull,则key转化为 unk

否则key保持不变
"""
def convert_key_by_count_dict(key, key_count_dict:dict, min_cnt=10, default_key="other", default_fill_na_value="unk"):
    cnt = key_count_dict.get(key)
    if pd.isnull(key) or cnt is None:
        normed_key = default_fill_na_value
    elif cnt <= min_cnt: # cnt太小
        normed_key = default_key
    else:
        normed_key = key
    #print("key:", key, "cnt:", cnt, " normed_key", normed_key)
    return normed_key

def get_normed_category_feature(ser:pd.Series, min_cnt=100, case_sensitive=False, default_value="other", default_fill_na_value="unk"):
    ser = ser.apply(lower_str_value, case_sensitive=case_sensitive)
    value_cnt_map = dict(ser.value_counts())
    print("value_cnt_map:", value_cnt_map)
    return ser.apply(convert_key_by_count_dict,
                     key_count_dict=value_cnt_map,
                     min_cnt=min_cnt,
                     default_key=default_value,
                     default_fill_na_value= default_fill_na_value)
    #ser.fillna(default_fill_na_value, inplace=True)

def column_missing_percent(x:pd.Series):
    return x.isnull().mean()

def preprocess(data:pd.DataFrame,
               feat_name, new_feat_name=None,
               lower_value=False,
               show_info=True,
               plot_fig = True
               ):
    print("process feature {} {}".format(feat_name, "."*100))
    if lower_value:
        print("lower value")
        data[feat_name]= data[feat_name].apply(lower_str_value, case_sensitive=False)
    if new_feat_name is not None:
        print("rename[{} -> {}]".format(feat_name, new_feat_name))
        data.rename(columns={feat_name: new_feat_name}, inplace=True)
        feat_name = new_feat_name
    if show_info:
        print("show stat info:")
        show_stat_info(data[feat_name], plot_fig=plot_fig)


# determine the threshold for missing values
def show_stat_info_of_all_feature(data:pd.DataFrame, plot_fig=True):
    df_cols = list(pd.DataFrame(data).columns)
    print("data types:", data.dtypes)
    print("data describe:", data.describe())
    print("columns size:{} columns:{}".format(len(df_cols), ",".join(df_cols)))
    for i in range(0, len(df_cols)):
        if df_cols[i].lower() != "user_id":
            show_stat_info(data[df_cols[i]], plot_fig=plot_fig)

def print_qcut_bins(ser_input:pd.Series, bin_size=10):
    ser = ser_input.dropna()
    cut_series, bins = pd.qcut(ser, q=bin_size, retbins= True, duplicates='drop')
    print("col", ser_input.name,"bins:",",".join(["%f" % x for x in bins]), " size:", len(bins))

def show_stat_info(ser_input:pd.Series, bin_size=10, plot_fig=True, is_category=False):
    ser = ser_input
    LEN = 50
    #print("stat col:"+ str(list(pd.DataFrame(ser_input).columns)))
    print("stat col:"+ ser_input.name+ " dtype:"+str(ser_input.dtype))
    print("="*LEN)
    get_and_print_missing_precent(pd.DataFrame(ser_input))
    print("="*LEN)
    if ser.dtype == object or is_category:
        value_cnt = ser.value_counts()
        print("describe:", ser.describe())
        print("="*LEN)
        print("value_cnt:", value_cnt)
        print("="*LEN)
        print("value_cnt histogram:", np.histogram(value_cnt, bins=bin_size))
        print("="*LEN)
        if plot_fig:
            print("plot value count hist:")
            value_cnt.hist()
            #print("qcut_cnt hist:", cut_series)
            print("="*LEN)
            print("plot value count hist by seaborn")
            plot_float_data_distribute(value_cnt, "value_cnt")
        try:
            print("="*LEN)
            cut_series, bins = pd.qcut(value_cnt, q= bin_size, retbins= True, duplicates='drop')
            print("col name", ser_input.name,"bins:",",".join(["%f" % x for x in bins]), " size:", len(bins))
            print("bin size:", bin_size, "qcut bins:", bins)
            print("value count after qcut:", cut_series.value_counts().head(50))
        except Exception as ex:
            print("qcut error for:", ser.name, ex)
        print("="*LEN)
    else:
        ser = ser_input.dropna()
        print("describe:", ser.describe())
        print("="*LEN)
        print("bin size:", bin_size, "histogram:", np.histogram(ser.dropna(), bins=bin_size))
        if plot_fig:
            print("="*LEN)
            print("plot hist")
            ser.hist(bins=bin_size, alpha=0.3, color='k', normed=True)
            ser.plot(kind='kde', style='k--')

        print("="*LEN)
        try:
            cut_series, bins = pd.qcut(ser, q=bin_size, retbins= True, duplicates='drop')
            print("col name", ser_input.name,"bins:",",".join(["%f" % x for x in bins]), " size:", len(bins))
            print("bin size:", bin_size, "qcut bins:", bins, " value count after qcut:", cut_series.value_counts)
        except Exception as ex:
            print("qcut error for:", ser.name, ex)
        if plot_fig:
            print("="*LEN)
            SAMPLE = min(10000, ser_input.size)
            print("plot distribute by seaborn, sample:", SAMPLE)
            plot_float_data_distribute(ser.sample(SAMPLE, replace=True), "value")

def clip_float_by_quantile(ser:pd.Series, min=0.02, max=0.98, div_max=True):
    v_min = ser.quantile(min)
    v_max = ser.quantile(max)
    cliped = ser.clip(lower=v_min, upper= v_max)
    if div_max:
        return cliped/(v_max+EPS)
    else:
        return cliped

def get_bucket_index(x, bucks):
    if x<bucks[0]:
        ind = 0
    elif x>bucks[-1]:
        ind = len(bucks)
    else:
        ind = np.where(bucks<=x)[0][-1]
    return ind

def show_box_plot_for_numberic_features(data:pd.DataFrame, cols:list):
    # Create box plots for all numeric features
    sns.set_style("white")
    f, ax = plt.subplots(figsize=(50, len(cols)))
    ax.set_xscale("log")
    ax = sns.boxplot(data=data[cols], orient="h", palette="Set1")
    ax.xaxis.grid(False)
    ax.set(ylabel="Feature names")
    ax.set(xlabel="Numeric values")
    ax.set(title="Numeric Distribution of Features")
    sns.despine(trim=True, left=True)

"""
We use the scipy function boxcox1p which computes the Box-Cox transformation. The goal is to find a simple transformation that lets us normalize data.
"""
def get_high_skew_features(data:pd.DataFrame, cols:list=None, skew_threshold=0.5):
    # Find skewed numerical features
    if cols is None:
        cols = get_all_numberic_columns(data)
    skew_features = data[cols].apply(lambda x: skew(x)).sort_values(ascending=False)
    print("skew features:{}".format(skew_features))

    high_skew = skew_features[skew_features > skew_threshold]
    high_skew_index = high_skew.index

    print("There are {} numerical features with Skew > {} :".format(high_skew.shape[0], skew_threshold))
    #skewness = pd.DataFrame({'Skew': high_skew})
    return high_skew_index

def boxcox1p_norm(data:pd.DataFrame, skew_col_indexs):
    # Normalize skewed features
    for col in skew_col_indexs: #
        print("boxcox col:", col)
        data[col] = boxcox1p(data[col], boxcox_normmax(data[col] + 1))
    print("boxcox1p norm done!")

"""
1.category用简单的mode填充
2.numberic类型用median填充
3.将string中的空格转成 _
"""
def fill_na_by_type(data:pd.DataFrame, exclude_cols=["user_id"]):
    for col in list(data.columns):
        if col in exclude_cols:
            continue
        if data[col].dtype == object:
            value=data[col].mode()[0]
            print("feat:{} fill na by mode:{}".format(col, value))
        else:
            value=data[col].median()
            print("feat:{} fill na by median:{}".format(col, value))
        #
        data[col].fillna(value, inplace=True)
        if data[col].dtype == object:
            data[col]=data[col].apply(lambda x:x.replace(" ", "_"))

def replace_space_value(data:pd.DataFrame, cols, new="_"):
    print("replace space with {} in cols {}".format(new, cols))
    for col in cols:
        if data[col].dtype == object:
            data[col]=data[col].apply(lambda x:x.replace(" ", new))

def remove_cols_if_exists(data:pd.DataFrame, cols:list):
    for col in cols:
        if col in list(data.columns):
            print("remove col:{}".format(col))
            data.drop([col], axis=1, inplace=True)

def common_process(data:pd.DataFrame):
    str_cols = get_all_str_columns(data)
    print("replace space")
    replace_space_value(data, str_cols)
    #data.set_index(user_id_name, inplace=True)
    #data.add_prefix()

def load_str_feat_cols(col_to_values_file):
    #col_to_values_file = "{}{}".format(DATA_WITH_FEATURE_PATH, "col_to_value.txt")
    print("read str col from path:", col_to_values_file)
    with open(col_to_values_file, "r") as f_obj:
        col_to_values = json.load(f_obj)
    return col_to_values

def save_str_feat_cols(str_col_to_values_map, col_to_values_file):
    with open(col_to_values_file, "w") as f_obj:
        json.dump(str_col_to_values_map, f_obj, ensure_ascii=False, indent=2)
    print("save str col to path:", col_to_values_file)

def save_numberic_feat_cols(float_values, float_values_file):
    with open(float_values_file, "w") as f_obj:
        json.dump(float_values, f_obj, ensure_ascii=False, indent=2)
    print("save float col to path:", float_values_file)

# { f1:[1,2,3], f2:[4,5,6] } => f1_1,f1_2,f1_3,f2_4,f2_5,f2_6
def col_value_map_to_col_list(col_value_map:dict, sep="_"):
    all_col_values = []
    for col in col_value_map.keys():
        all_col_values.extend([col+sep+value for value in col_value_map[col]])
    return all_col_values

def get_time_str():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def get_simple_time_str():
    return strftime("%Y%m%d_%H_%M_%S", localtime())

def performance_report(y, pred_y):
    print("pred_y shape:", pred_y.shape) # class:0, 1,
    auc_score = roc_auc_score(y, pred_y)
    print("auc:", auc_score)
    pred_label = (pred_y > 0.5)*1
    #precision, recall, thresholds = precision_recall_curve(test_y, pred_label)
    print("{} classification report:".format(get_time_str()))
    print(classification_report(y, pred_label))
    return auc_score

"""
def performance_report(test_y, test_y_pred, train_y, train_y_pred):
    print("="*80)
    print("train data preformance:")
    performance_report(train_y, train_y_pred)
    print("="*80)
    print("test data preformance:")
    performance_report(test_y, test_y_pred)
"""

def split_or_load_data(out_feated_data_path, all_feat_data, rand_state,
                       force_load_data=False, test_size = 0.3, repeat_pos_num=0,
                       use_full_data_to_train=False,
                       user_id_name="user_id",
                       label_name="label"
                       ):
    input_feat_data = out_feated_data_path+all_feat_data
    train_x_path = out_feated_data_path+"train_x.csv"
    test_x_path = out_feated_data_path+"test_x.csv"
    train_y_path = out_feated_data_path+"train_y.csv"
    test_y_path = out_feated_data_path+"test_y.csv"

    print("force_load_data:", force_load_data)
    if force_load_data \
            or not os.path.exists(train_x_path) \
            or not os.path.exists(test_x_path) \
            or not os.path.exists(train_y_path) \
            or not os.path.exists(test_y_path) \
            :
        print("split data from full data from path:{} ...".format(input_feat_data))
        #data = pd.read_csv(input_feat_data, encoding="utf-8", nrows=10000)
        data = pd.read_csv(input_feat_data, encoding="utf-8")
        train_x, test_x, train_y, test_y = train_test_split(data.drop([label_name], axis=1),
                                                            data[[user_id_name,label_name]], #将user_id也存下来
                                                            shuffle=True,
                                                            test_size=test_size,
                                                            random_state=rand_state)
        print("save split data to path:{}".format([train_x_path, test_x_path, train_y_path, test_y_path]))
        save_data_to_csv(train_x, train_x_path)
        save_data_to_csv(test_x, test_x_path)
        save_data_to_csv(train_y, train_y_path)
        save_data_to_csv(test_y, test_y_path)
    else:
        print("load data from disk, {}".format([train_x_path, test_x_path, train_y_path, test_y_path]))
        train_x = pd.read_csv(train_x_path, encoding="utf-8")
        test_x = pd.read_csv(test_x_path, encoding="utf-8")
        train_y = pd.read_csv(train_y_path, encoding="utf-8")
        test_y = pd.read_csv(test_y_path, encoding="utf-8")

    remove_cols_if_exists(train_x, [user_id_name])
    remove_cols_if_exists(test_x, [user_id_name])
    remove_cols_if_exists(train_y, [user_id_name])
    remove_cols_if_exists(test_y, [user_id_name])

    print('use_full_data_to_train flag:{}'.format(use_full_data_to_train))
    if use_full_data_to_train:
        train_x = pd.concat([train_x, test_x], axis=0)
        train_y = pd.concat([train_y, test_y], axis=0)
        print("concat full data, train_x shape:{} train_y shape:{}".format(train_x.shape, train_y.shape))

    # 人为改变训练样本中样本中负样本的分布，即复制多份正样本
    train_x, train_y = repeat_x_y(train_x, train_y, repeat_pos_num=repeat_pos_num)
    # ----------
    show_label_info(train_y, name="train_y")
    train_y = np.array(train_y).ravel()

    if not use_full_data_to_train:
        show_label_info(test_y, name="test_y")
        test_y = np.array(test_y).ravel()

    print("train_x:{} test_x:{} train_y:{} test_y:{}".format(train_x.shape, test_x.shape, train_y.shape, test_y.shape))
    return train_x, test_x, train_y, test_y

def repeat_x_y(train_x:pd.DataFrame,train_y:pd.DataFrame, repeat_pos_num:int, label_name="label"):
    if repeat_pos_num > 1:
        print("REPEAT_NUM:{}".format(repeat_pos_num))
        pos_index = train_y[train_y[label_name] == 1.0].index
        rep_train_y = train_y.loc[pos_index.repeat(repeat_pos_num - 1)].reset_index(drop=True)
        train_y = pd.concat([train_y, rep_train_y], axis=0)

        rep_train_x = train_x.loc[pos_index.repeat(repeat_pos_num - 1)].reset_index(drop=True)
        train_x = pd.concat([train_x, rep_train_x], axis=0)

        print("replicate positive sample, repeat:{} origin pos:{} after:\n{}".format(
            repeat_pos_num, len(pos_index), train_y[label_name].value_counts()))
        # permute
        np.random.seed(0)
        permute = np.random.permutation(train_x.shape[0])
        train_x = train_x.take(permute)
        train_y = train_y.take(permute)
    else:
        print("REPEAT_NUM:{} <=1, not repeat".format(repeat_pos_num))
    return train_x, train_y

def get_auc(test_y, pred_y):
    fpr,tpr,_ = roc_curve(test_y, pred_y)
    roc_auc = auc(fpr,tpr)
    return roc_auc

def get_logger():
    import logging
    import sys
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler and set level to info
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # create formatter and add formatter to ch
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)  # add ch to the logger
    return logger

"""
libsvm format:
label feat_id:feat_value 
要求每个样本中feat_id升序排列
0 12:1 14:0.5 15:0.2
1 12:1 14:0.5 15:0.2
0 12:1 14:0.5 15:0.2
0 12:1 14:0.5 15:0.2
"""
def load_svm_file(data_file:str, n_features=None):
    from sklearn.datasets import load_svmlight_file
    train_data = load_svmlight_file(data_file, n_features=n_features)
    X = train_data[0].toarray() # array
    y = train_data[1]
    return X, y

# 将每个样本的特征按id升序排列
def sort_svm_feat(in_file, out_file):
    with open(out_file, "w") as fo:
        out_list= []
        with open(in_file, "r") as fr:
            for line in fr.readlines():
                arr = line.strip("\n").split(" ")
                label = arr[0]
                #feat_value =[(k,v) for (k,v) in arr[1:].split(":")]
                feats= []
                for feat in  arr[1:]:
                    f_arr = feat.split(":")
                    feats.append((int(f_arr[0]), f_arr[1]))
                feats = sorted(feats, key=lambda x:x[0])
                out = label+" "+" ".join(["{}:{}".format(k,v) for (k,v) in feats])
                out_list.append(out)
        fo.write("\n".join(out_list))

def write_str_to_file(feat_str, out_feat_file):
    with open(out_feat_file, "w") as f:
        f.write(feat_str)
    printf("out file:", out_feat_file)
