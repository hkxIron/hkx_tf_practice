import tensorflow as tf
from multiprocessing import Process

mydevice = "/cpu:0"
#mydevice = "/gpu:0"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)

mrange = 1000


def myfun():
    with tf.device(mydevice):
        mm1 = tf.constant([[float(i) for i in range(mrange)]], dtype='float32')
        mm2 = tf.constant([[float(i)] for i in range(mrange)], dtype='float32')

    with tf.device(mydevice):
        prod = tf.matmul(mm1, mm2)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
    rest = sess.run(prod)

    print(rest)
    sess.close()

if __name__ == "__main__":
    ll = []
    for i in range(5):
        p1 = Process(target=myfun)
        p1.start()
        ll.append(p1)

    for item in ll:
        item.join()
    print("main end")