import tensorflow as tf
import glob
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# file_name="/data2/kexinhu/online/increment/train/2018032906/non_compress/compress/part-r-00000.tfr.gz"
# writer = tf.python_io.TFRecordWriter(tf_filename, tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
file_name="/home/hukexin/work/dataset/mi_match/data/test/date=20180831/part-r-00097"
count = 0
break_count = 1
is_gzip = False

if is_gzip:
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
else:
    options = None

record_iterator = tf.python_io.tf_record_iterator(file_name, options)
for example in record_iterator:
    result = tf.train.Example.FromString(example)
    print(result)
    #print(result.features.feature['label'].int64_list.value)
    # print(result.features.feature['text_label'].bytes_list.value)
    count += 1
    if break_count >= 0 and count >= break_count: break
print("count:", count)