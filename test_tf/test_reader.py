#-*- coding:utf-8 -*-
import tensorflow as tf

def one_reader_one_sample():
    # 生成一个先入先出队列和一个QueueRunner,生成文件名队列
    filenames = [("data/"+x) for x in ['a.csv', 'b.csv', 'c.csv'] ]
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    # 定义Reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # 定义Decoder
    example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
    #example_batch, label_batch = tf.train.shuffle_batch([example,label], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)
    # 这里没有使用tf.train.shuffle_batch，会导致生成的样本和label之间对应不上，输出乱序
    # 运行Graph
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
        for i in range(10):
            print(example.eval(),label.eval())
        coord.request_stop()
        coord.join(threads)

def one_reader_shuffle_batch():
    # 生成一个先入先出队列和一个QueueRunner,生成文件名队列
    filenames = [("data/"+x) for x in ['a.csv', 'b.csv', 'c.csv'] ]
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    # 定义Reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # 定义Decoder
    example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
    example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size=5, capacity=200,
                                                        min_after_dequeue=100, num_threads=2)
    # 运行Graph
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
        for i in range(10):
            e_val, l_val = sess.run([example_batch, label_batch])
            print(e_val, l_val,"\n")
        coord.request_stop()
        coord.join(threads)


def one_reader_multi_sample():
    filenames =[("data/"+x) for x in ['a.csv', 'b.csv', 'c.csv'] ]
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    example, label = tf.decode_csv(value, record_defaults=[['null'], ['null']])
    # 使用tf.train.batch()会多加了一个样本队列和一个QueueRunner。
    #Decoder解后数据会进入这个队列，再批量出队。
    # 虽然这里只有一个Reader，但可以设置多线程，相应增加线程数会提高读取速度，但并不是线程越多越好。
    example_batch, label_batch = tf.train.batch(
          [example, label], batch_size=5)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            e_val,l_val = sess.run([example_batch,label_batch])
            print(e_val,l_val)
        coord.request_stop()
        coord.join(threads)

def multi_reader_multi_sample():
    filenames =[("data/"+x) for x in ['a.csv', 'b.csv', 'c.csv'] ]
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [['null'], ['null']]
    # 定义了多种解码器,每个解码器跟一个reader相连
    example_list = [tf.decode_csv(value, record_defaults=record_defaults)
                    for _ in range(10)]  # Reader设置为2
    # 使用tf.train.batch_join()，可以使用多个reader，并行读取数据。每个Reader使用一个线程。
    example_batch, label_batch = tf.train.batch_join(
        example_list, batch_size=5)
    print(example_batch)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            e_val, l_val = sess.run([example_batch, label_batch])
            print(e_val, l_val,"\n")
        coord.request_stop()
        coord.join(threads)

def read_feat():
    # 生成一个先入先出队列和一个QueueRunner,生成文件名队列
    filenames = ["data/feat.csv"]
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    # 定义Reader
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    # 定义Decoder
    record_defaults = [[1], [1], [1], [1], [1]]
    col1, col2, col3, col4, col5 = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.pack([col1, col2, col3])
    label = tf.pack([col4, col5])
    example_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=2, capacity=200,
                                                        min_after_dequeue=100, num_threads=2)
    # 运行Graph
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)  # 启动QueueRunner, 此时文件名队列已经进队。
        for i in range(10):
            e_val, l_val = sess.run([example_batch, label_batch])
            print(e_val, l_val)
        coord.request_stop()
        coord.join(threads)

def main():
    #one_reader_one_sample()
    #one_reader_shuffle_batch()
    #one_reader_multi_sample()
    multi_reader_multi_sample()

if __name__ == "__main__":
   main()