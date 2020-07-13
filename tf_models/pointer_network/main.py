import sys
import numpy as np
import tensorflow as tf

from config import get_config
from utils import prepare_dirs_and_logger, save_config
from data_util import gen_data
from model import Model

config = None

def main(_):
    prepare_dirs_and_logger(config)

    if not config.task.lower().startswith('tsp'):
        raise Exception("[!] Task should starts with TSP")

    if config.max_enc_length is None:
        config.max_enc_length = config.max_data_length # 10
    if config.max_dec_length is None:
        config.max_dec_length = config.max_data_length # 10

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    model = Model(config)

    batch_size = config.batch_size


    print("train data dir:", config.train_data_dir)
    train_enc_seq, train_target_seq, train_enc_seq_length, train_target_seq_length = gen_data(config.train_data_dir)

    # 最后一个batch为eval集
    eval_enc_seq,eval_target_seq,eval_enc_seq_length,eval_target_seq_length = train_enc_seq[-batch_size:], \
                                                                              train_target_seq[-batch_size:], \
                                                                              train_enc_seq_length[-batch_size:], \
                                                                              train_target_seq_length[-batch_size:]
    #0~ -1*batch的所有数据均作为训练用
    train_enc_seq, train_target_seq, train_enc_seq_length, train_target_seq_length = train_enc_seq[: -batch_size], \
                                                                                  train_target_seq[:-batch_size], \
                                                                                  train_enc_seq_length[:-batch_size], \
                                                                                  train_target_seq_length[:-batch_size]

    test_enc_seq, test_target_seq, test_enc_seq_length, test_target_seq_length = gen_data(config.test_data_dir)

    batch_num = len(train_enc_seq) // batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(min(config.max_step, batch_num)):
            begin_index = step*batch_size
            end_index = (step+1)*batch_size
            train_batch={
                'enc_seq': train_enc_seq[begin_index:end_index],
                'enc_seq_length': train_enc_seq_length[begin_index:end_index],
                'target_seq': train_target_seq[begin_index:end_index],
                'target_seq_length': train_target_seq_length[begin_index:end_index]
            }
            loss, debug_info = model.train(sess, train_batch)
            print(str(step) + " train loss : " + str(loss))

            if step > 0 and step % config.eval_step == 0:
                print("begin to eval...")
                eval_batch = {
                    'enc_seq': eval_enc_seq,
                    'enc_seq_length': eval_enc_seq_length,
                    'target_seq': eval_target_seq,
                    'target_seq_length': eval_target_seq_length
                }
                eval_loss = model.eval(sess, eval_batch)
                print(str(step) + " eval loss : " + str(eval_loss))

if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)