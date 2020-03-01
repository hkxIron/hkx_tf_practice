import tensorflow as tf
import os
import sys
import pandas as pd
import tensorflow as tf


def test_numeric_origin():
    FUTURES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']

    # 格式化数据文件的目录地址
    dir_path = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(dir_path, 'iris_train.csv')
    test_path = os.path.join(dir_path, 'iris_test.csv')

    # 载入训练数据
    train = pd.read_csv(train_path, names=FUTURES, header=0)
    train_x, train_y = train, train.pop('Species')

    # 载入测试数据
    test = pd.read_csv(test_path, names=FUTURES, header=0)
    test_x, test_y = test, test.pop('Species')

    # 拼合特征列
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    print(feature_columns)
"""
[_NumericColumn(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), _NumericColumn(key='SepalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), _NumericColumn(key='PetalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None), _NumericColumn(key='PetalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]
"""


def test_numeric():
    price = {'price': [[1.],
                       [2.],
                       [3.],
                       [4.]]}  # 4行样本

    column = tf.feature_column.numeric_column('price', normalizer_fn=lambda x:x+2)
    tensor = tf.feature_column.input_layer(price,[column]) #

    with tf.Session() as session:
        print(session.run([tensor]))
"""
[array([[3.],
       [4.],
       [5.],
       [6.]], dtype=float32)]
"""


def test_bucket():
    years = {'years': [1999, 2013, 1987, 2005]}

    years_fc = tf.feature_column.numeric_column('years')
    column = tf.feature_column.bucketized_column(years_fc, [1990, 2000, 2010])
    tensor = tf.feature_column.input_layer(years, [column])

    with tf.Session() as session:
        print(session.run([tensor]))
"""
[array([[0., 1., 0., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.]], dtype=float32)]

如果没有input_layer,那么bucketized_column输出的是index:
[array([[1],
       [3],
       [0],
       [2]], dtype=float32)]
"""

def test_multi_column():
    years = {'years': [1999, 2013, 1987, 2005]} # 就是模拟的 tfrecord里的kv值

    years_fc = tf.feature_column.numeric_column('years')
    year_column = tf.feature_column.bucketized_column(years_fc, [1990, 2000, 2010])

    pets = {'pets': [2, 3, 0, 1]}  # 猫0，狗1，兔子2，猪3
    pets_col = tf.feature_column.categorical_column_with_identity(key='pets', num_buckets=4)
    pets_column = tf.feature_column.indicator_column(pets_col) # category column必须要经过indicator 或者embedding
    pets_column = pets_col

    all_feats = {**years, **pets}
    tensor_year = tf.feature_column.input_layer(years, [year_column])
    tensor_pet = tf.feature_column.input_layer(pets, [pets_column])
    tensor_year_pet = tf.feature_column.input_layer(all_feats, [year_column, pets_column]) # 将多列同时映射成one-hot

    with tf.Session() as session:
        print("year:", session.run([tensor_year]))
        print("pet:", session.run([tensor_pet]))
        print("year pet:", session.run([tensor_year_pet]))

"""
year: 
[array([[0., 1., 0., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 0., 1., 0.]], dtype=float32)]
pet: 
[array([[0., 0., 1., 0.],
       [0., 0., 0., 1.],
       [1., 0., 0., 0.],
       [0., 1., 0., 0.]], dtype=float32)]
year pet: 
[array([[0., 0., 1., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 1.],
       [1., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 1., 0.]], dtype=float32)]
"""
    #sys.exit(-1)

def test_indicator():
    import tensorflow as tf

    pets = {'pets': [2, 3, 0, 1]}  # 猫0，狗1，兔子2，猪3

    column = tf.feature_column.categorical_column_with_identity(
        key='pets',
        num_buckets=4)

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])

    with tf.Session() as session:
        print(session.run([tensor]))
        """
        [array([[0., 0., 1., 0.], #兔子
        [0., 0., 0., 1.], #猪
        [1., 0., 0., 0.], #猫
        [0., 1., 0., 0.]], dtype=float32)] #狗
        """
def test_category_0():
    import tensorflow as tf
    pets = {'pets': ['rabbit', 'pig', 'dog', 'mouse', 'cat']}
    column = tf.feature_column.categorical_column_with_vocabulary_list( # category只能映射到 0~n之间的数字,但并非one-hot
        key='pets',
        vocabulary_list=['cat', 'dog', 'rabbit', 'pig'],
        dtype=tf.string,
        default_value=-1,
        num_oov_buckets=3) # 额外增加的bucket

    tensor = tf.feature_column.input_layer(pets, [column]) # linear

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
    sys.exit(0)

def test_category():
    import tensorflow as tf
    pets = {'pets': ['rabbit', 'pig', 'dog', 'mouse', 'cat']}
    column = tf.feature_column.categorical_column_with_vocabulary_list( # category只能映射到 0~n之间的数字,但并非one-hot
        key='pets',
        vocabulary_list=['cat', 'dog', 'rabbit', 'pig'],
        dtype=tf.string,
        default_value=-1,
        num_oov_buckets=3) # 额外增加的bucket

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
    """
    [array([[0., 0., 1., 0., 0., 0., 0.], #'rabbit'
       [0., 0., 0., 1., 0., 0., 0.], #'pig'
       [0., 1., 0., 0., 0., 0., 0.], #'dog'
       [0., 0., 0., 0., 0., 1., 0.], #mouse
       [1., 0., 0., 0., 0., 0., 0.]], dtype=float32)] #'cat'
    """

def test_category_file():
    import os
    import tensorflow as tf

    pets = {'pets': ['rabbit', 'pig', 'dog', 'mouse', 'cat']}

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fc_path = os.path.join(dir_path, 'pets_fc.txt')

    column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="pets",
        vocabulary_file=fc_path,
        num_oov_buckets=0)

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
        """
        [array([[0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [0., 1., 0., 0.],
        [0., 0., 0., 0.], # mouse全0
        [1., 0., 0., 0.]], dtype=float32)]
        """

def test_hash():
    import tensorflow as tf

    colors = {'colors': ['green', 'red', 'blue', 'yellow', 'pink', 'blue', 'red', 'indigo']}

    column = tf.feature_column.categorical_column_with_hash_bucket(
        key='colors',
        hash_bucket_size=5, # 一般bucket size是值数量的5~10倍
    )

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(colors, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
"""
[array([[0., 0., 0., 0., 1.], #green
       [1., 0., 0., 0., 0.], # red
       [1., 0., 0., 0., 0.], # blue, red与blue的hash值一样
       [0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [1., 0., 0., 0., 0.], # red
       [1., 0., 0., 0., 0.], # blue
       [0., 1., 0., 0., 0.]], dtype=float32)]
"""


def test_cross_feature():
    import tensorflow as tf

    featrues = {
        'longtitude': [19, 61, 30, 9, 45],
        'latitude': [45, 40, 72, 81, 24]
    }

    longtitude = tf.feature_column.numeric_column('longtitude')
    latitude = tf.feature_column.numeric_column('latitude')

    longtitude_b_c = tf.feature_column.bucketized_column(longtitude, boundaries=[33, 66])
    latitude_b_c = tf.feature_column.bucketized_column(latitude, boundaries=[33, 66])

    cross_column = tf.feature_column.crossed_column([longtitude_b_c, latitude_b_c], hash_bucket_size=12)

    indicator = tf.feature_column.indicator_column(cross_column) # indicator就是变为one-hot or multi-hot
    tensor = tf.feature_column.input_layer(featrues, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("cross feature:")
        print(session.run([tensor]))
"""
[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]], dtype=float32)]
"""

def test_embed():
    # 参考公式: embedding_dim = sqrt(sqrt(vocab_size))
    import tensorflow as tf

    features = {'pets': ['dog', 'cat', 'rabbit', 'pig', 'mouse']}

    pets_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
        'pets',
        ['cat', 'dog', 'rabbit', 'pig'],
        dtype=tf.string,
        default_value=-1)

    column = tf.feature_column.embedding_column(pets_f_c, dimension=3)
    tensor = tf.feature_column.input_layer(features, [column])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))

"""
[array([[ 0.10850643,  0.41479287, -0.30671582],
       [-0.24762067,  0.70139533,  0.454066  ],
       [-0.07910448, -0.43386784, -0.2662065 ],
       [ 0.07516848, -0.59056175, -0.24181622],
       [ 0.        ,  0.        ,  0.        ]], dtype=float32)]
"""

def test_weighted_category_column():
    import tensorflow as tf
    from tensorflow.python.feature_column.feature_column import _LazyBuilder

    features = {'color': [['R'], ['A'], ['G'], ['B'], ['R']],
                'weight': [[1.0], [5.0], [4.0], [8.0], [3.0]]}

    color_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
        key='color',
        vocabulary_list = ['R', 'G', 'B', 'A'],
        dtype = tf.string,
        default_value = -1)

    weighted_column = tf.feature_column.weighted_categorical_column(color_f_c, weight_feature_key='weight')
    indicator = tf.feature_column.indicator_column(weighted_column) # one-hot
    tensor = tf.feature_column.input_layer(features, feature_columns=[indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
"""
运行之后得到下面输出，权重改变了独热模式，不仅包含0或1，还带有权重值
[array([[1., 0., 0., 0.],
       [0., 0., 0., 5.],
       [0., 4., 0., 0.],
       [0., 0., 8., 0.],
       [3., 0., 0., 0.]], dtype=float32)]
"""


def test_linear_model():
    import tensorflow as tf
    from tensorflow.python.feature_column.feature_column import _LazyBuilder

    def get_linear_model_bias():
        with tf.variable_scope('linear_model', reuse=True):
            return tf.get_variable('bias_weights')

    def get_linear_model_column_var(column):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 'linear_model/' + column.name)[0]

    # 共有3个样本
    featrues = {
        'price': [[1.0], [5.0], [10.0]],
        'color': [['R'], ['G'], ['B']]
    }

    price_column = tf.feature_column.numeric_column('price')
    color_column = tf.feature_column.categorical_column_with_vocabulary_list('color',
                                                                             ['R', 'G', 'B'])
    prediction = tf.feature_column.linear_model(featrues, feature_columns=[price_column, color_column])

    bias = get_linear_model_bias()
    price_var = get_linear_model_column_var(price_column) # w_price:[1,]
    color_var = get_linear_model_column_var(color_column) # w_color:[3,1]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        sess.run(bias.assign([7.0])) # bias = 7
        sess.run(price_var.assign([[10.0]])) # w_price = 10
        sess.run(color_var.assign([[2.0], [2.0], [2.0]])) # w_color =[2,2,2]

        predication_result = sess.run([prediction])

        print(prediction)
        #对于第一个样本 w_price*price + w_color^T*color + bias = 10*1+[2,2,2]^T*[1,0,0]+7=10+2+7=19
        print(predication_result)
"""
对所有特征进行线性加权操作（数值和权重值相乘）

Tensor("linear_model_1/linear_model/weighted_sum:0", shape=(3, 1), dtype=float32)
[array([[ 19.], #对于第一个样本 w_price*price + w_color^T*color + bias = 10*1+[2,2,2]^T*[1,0,0]+7=10+2+7=19
       [ 59.],
       [109.]], dtype=float32)]
"""


"""
线性分类器 linearClassifier和线性回归器linearRegressor，接收所有类型特征列；
深度神经网络分类器DNNClassifier和深度神经网络回归器DNNRegressor，仅接收密集特征列dense column,其他类型特征列必须用指示列indicatorColumn或嵌入列embedingColumn进行包裹
线性神经网络合成分类器linearDNNCombinedClassifier和线性神经网络合成回归器linearDNNCombinedRegressor：
linear_feature_columns参数接收所有类型特征列
dnn_feature_columns只接收密度特征列dense column
"""

test_multi_column()
test_bucket()
#test_category_0()
test_numeric_origin()
test_numeric()
test_indicator()
test_category()
test_category_file()
test_hash()
test_cross_feature()
test_embed()
test_weighted_category_column()
test_linear_model()


