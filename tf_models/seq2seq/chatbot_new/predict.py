import tensorflow as tf
from data_helpers import loadDataset, getAllTrainBatches, sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np


tf.app.flags.DEFINE_integer('rnn_size', 100, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)
print("some smaples:", trainingSamples[:3])

def predict_ids_to_seq(predict_ids, id2word, beam_size):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: [batch,decode_len,beam_size]的ndarray数组
    :param id2word: vocab字典
    :return:
    '''
    #for single_predict in predict_ids:
    for i in range(beam_size):
        predict_list = np.ndarray.tolist(predict_ids[:, :, i])
        predict_seq = [id2word[idx] for idx in predict_list[0]] #由于batch=1,所以取第0个元素即可
        print(" ".join(predict_seq))

with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='decode', use_attention=True, beam_search=True, beam_size=5, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..', ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        batch = sentence2enco(sentence, word2id)
        [predicted_ids] = model.infer(sess, batch)
        print("predicted_ids:", predicted_ids)
        print("reply words:")
        predict_ids_to_seq(predicted_ids, id2word, beam_size=4)
        print("> ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()