
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _parse_func(example):
    features = {"x": tf.FixedLenFeature((1,), tf.float32),
                "y": tf.FixedLenFeature((1,), tf.float32)}
    parsed_features = tf.parse_single_example(example, features)
    return parsed_features["x"], parsed_features["y"]


# In[12]:


def model(x, y):
    w = tf.Variable([0.3])
    b = tf.Variable([0.3])
    linear_model = w * x + b
    loss = tf.nn.l2_loss(tf.to_float(y) - w * tf.to_float(x) - b)
#     loss = tf.reduce_sum(tf.square(y - linear_model))
    return loss, w, b


# In[8]:


# write
filename = 'd:/temp.tfr.gzip'
option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
writer = tf.python_io.TFRecordWriter(filename, options=option)
for i in range(0, 1000):
    x = 1 + i * 0.001
    y = x + 1
    example = tf.train.Example(features=tf.train.Features(feature={
        "x": _float_feature(x),
        "y": _float_feature(y)
    }))
    writer.write(example.SerializeToString())
writer.close()


# In[13]:


dataset = tf.data.TFRecordDataset([filename], compression_type="GZIP")
iterator = dataset.map(_parse_func).make_one_shot_iterator()


# In[14]:


x, y = iterator.get_next()
loss, w, b = model(x, y)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


# In[15]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run([w, b]))
for i in range(0, 1000):
    sess.run(train_step)
    print(sess.run([w, b]))


# In[9]:


sess.run(loss)

