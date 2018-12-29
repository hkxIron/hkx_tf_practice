import numpy as np
import pickle

direct_encoding_fields = ['hour', 'C1', 'C15', 'C16', 'C18', 'C20',
                          'banner_pos',  'site_category','app_category',
                          'device_type','device_conn_type']

frequency_encoding_fields = ['C14','C17', 'C19', 'C21',
                             'site_id','site_domain','app_id','app_domain',
                              'device_model', 'device_id']

dataPath = "../data/"
# load direct encoding fields
with open(dataPath+'sets/click.pkl','rb') as f:
    click = pickle.load(f)

with open(dataPath+'sets/hour.pkl','rb') as f:
    hour = pickle.load(f)

with open(dataPath+'sets/C1.pkl','rb') as f:
    C1 = pickle.load(f)

with open(dataPath+'sets/C15.pkl','rb') as f:
    C15 = pickle.load(f)

with open(dataPath+'sets/C16.pkl','rb') as f:
    C16 = pickle.load(f)

with open(dataPath+'sets/C18.pkl','rb') as f:
    C18 = pickle.load(f)

with open(dataPath+'sets/C20.pkl','rb') as f:
    C20 = pickle.load(f)

with open(dataPath+'sets/banner_pos.pkl','rb') as f:
    banner_pos = pickle.load(f)

with open(dataPath+'sets/site_category.pkl','rb') as f:
    site_category = pickle.load(f)

with open(dataPath+'sets/app_category.pkl','rb') as f:
    app_category = pickle.load(f)

with open(dataPath+'sets/device_type.pkl','rb') as f:
    device_type = pickle.load(f)

with open(dataPath+'sets/device_conn_type.pkl','rb') as f:
    device_conn_type = pickle.load(f)


# loading frequency encoding fields
# field2count dictionaries
with open(dataPath+'field2count/C14.pkl','rb') as f:
    C14 = pickle.load(f)

with open(dataPath+'field2count/C17.pkl','rb') as f:
    C17 = pickle.load(f)

with open(dataPath+'field2count/C19.pkl','rb') as f:
    C19 = pickle.load(f)

with open(dataPath+'field2count/C21.pkl','rb') as f:
    C21 = pickle.load(f)

with open(dataPath+'field2count/site_id.pkl','rb') as f:
    site_id = pickle.load(f)

with open(dataPath+'field2count/site_domain.pkl','rb') as f:
    site_domain = pickle.load(f)

with open(dataPath+'field2count/app_id.pkl','rb') as f:
    app_id = pickle.load(f)

with open(dataPath+'field2count/app_domain.pkl','rb') as f:
    app_domain = pickle.load(f)

with open(dataPath+'field2count/device_model.pkl','rb') as f:
    device_model = pickle.load(f)

with open(dataPath+'field2count/device_id.pkl','rb') as f:
    device_id = pickle.load(f)


ind = 0
# 按照每个变量取值的个数,进行one-hot编码
for field in direct_encoding_fields:
    # value to one-hot-encoding index dict
    field_dict = {}
    field_sets = eval(field)
    for value in list(field_sets):
        field_dict[value] = ind
        ind += 1
    with open(dataPath+'dicts/'+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)

# value类特征,应该按照值的范围,分段线性one-hot
# 但作者在此处并没有采取此种方法, 而是将值进行统计,感觉泛化性能不行啊
for field in frequency_encoding_fields:
    # value to one-hot-encoding index dict
    field_dict = {}
    field2count = eval(field)
    index_rare = ind
    for k,count in field2count.items(): # value->value_count
        if count < 10:
            field_dict[k] = index_rare
        else:
            field_dict[k] = ind + 1
            ind += 1

    with open(dataPath+'dicts/'+field+'.pkl', 'wb') as f:
        pickle.dump(field_dict, f)


field_dict = {}
field_sets = click
for value in list(field_sets):
    field_dict[value] = ind + 1
    ind += 1
with open(dataPath+'dicts/'+'click'+'.pkl', 'wb') as f:
    pickle.dump(field_dict, f)





