import tensorflow as tf
import codecs

BATCH_SIZE = 6
EXPOCH_SIZE = 5

def input_producer():
    #array = codecs.open("data/test.txt").readlines()
    #print("array:",array)
    #array = map(lambda line: line.strip(), array)
    #array =[int(x.strip()) for x in array]
    array = [i for i in range(1,36)]
    print("array:",array)
    i = tf.train.range_input_producer(EXPOCH_SIZE, num_epochs=1, shuffle=False).dequeue()
    inputs = tf.slice(array, begin=[i * BATCH_SIZE], size=[BATCH_SIZE]) # 注意 后面是size而不是end
    return inputs


class Inputs(object):
    def __init__(self):
        self.inputs = input_producer()


def main(*args, **kwargs):
    inputs = Inputs()
    # init = tf.group(tf.initialize_all_variables(),
    #                 tf.initialize_local_variables())
    init =tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        index = 0
        while not coord.should_stop() and index < 10:
            datalines = sess.run(inputs.inputs)
            index += 1
            print("step: %d, batch data: %s" % (index, str(datalines)))
    except tf.errors.OutOfRangeError:
        print("Done traing:-------Epoch limit reached")
    except KeyboardInterrupt:
        print("keyboard interrput detected, stop training")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    del sess


if __name__ == "__main__":
    main()