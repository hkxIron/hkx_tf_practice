# chat_bot_seq2seq_attention
# TODO: inference也有问题,batch维度貌似不对

# blog: https://www.jianshu.com/p/aab40f439012
# github: https://github.com/princewen/tensorflow_practice/blob/master/nlp/chat_bot_seq2seq_attention/train.py

import tensorflow as tf

from data_loader import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os

# http://blog.csdn.net/leiting_imecas/article/details/72367937
# tf定义了tf.app.flags，用于支持接受命令行传递参数，相当于接受argv。
tf.app.flags.DEFINE_integer('rnn_size', 100, '1024, Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 20, '1024, Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 20, '128, Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 500, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)


with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size,
                         FLAGS.num_layers,
                         FLAGS.embedding_size,
                         FLAGS.learning_rate,
                         word2id,
                         mode='train',
                         use_attention=True,
                         use_beam_search=True,
                         beam_size=5,
                         max_gradient_norm=5.0
                         )

    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #或者latest_model = tf.train.latest_checkpoint(FLAGS.model_dir)
        latest_model = ckpt.model_checkpoint_path
        print('Reloading model parameters, latest_model:'+latest_model)
        model.saver.restore(sess, latest_model)
    else:
        print('Created new model parameters..')
        sess.run(tf.global_variables_initializer())

    #current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        batches = getBatches(trainingSamples, FLAGS.batch_size)
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        for nextBatch in tqdm(batches, desc="Training"):
            loss, summary, global_step = model.train(sess, nextBatch)
            #current_step += 1
            # 每多少步进行一次保存
            if global_step % FLAGS.steps_per_checkpoint == 0:
                # 由于是交叉熵loss,所以可以计算
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (global_step, loss, perplexity))
                summary_writer.add_summary(summary, global_step)
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=global_step)

