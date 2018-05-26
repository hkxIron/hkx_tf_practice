import glob
import tensorflow as tf
import cPickle
import numpy as np
from tqdm import tqdm

import collections
from multiprocessing import Process, Manager, Pool



def get_multihot_encoding(example_label):
    enc = np.zeros(10)
    for label in example_label:
        if label in lookup.values():
            index = lookup_inverted[label]
            enc[index] = 1
    return list(enc)


# Set-up MultiProcessing
manager = Manager()
audio_embeddings_dict = manager.dict()
audio_labels_dict = manager.dict()
audio_multihot_dict = manager.dict()
sess = tf.Session()

# The iterable which gets passed to the function
all_tfrecord_filenames = glob.glob('/Users/jeff/features/audioset_v1_embeddings/unbal_train/*.tfrecord')



def process_tfrecord(tfrecord):

  for idx, example in enumerate(tf.python_io.tf_record_iterator(tfrecord)):
    tf_example = tf.train.Example.FromString(example)
    vid_id = tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
    example_label = list(np.asarray(tf_example.features.feature['labels'].int64_list.value))

    # Non zero intersect of 2 sets is True - only create dict entries if this is true!
    if set(example_label) & label_filters:
        print(set(example_label) & label_filters, " Is the intersection of the two")
        tf_seq_example = tf.train.SequenceExample.FromString(example)
        n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
        audio_frame = []

        for i in range(n_frames):
            audio_frame.append(tf.cast(tf.decode_raw(
                 tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],tf.uint8)
                ,tf.float32).eval(session=sess))

        audio_embeddings_dict[vid_id] = audio_frame
        audio_labels_dict[vid_id] = example_label
        audio_multihot_dict[vid_id] = get_multihot_encoding(example_label)
        #print(get_multihot_encoding(example_label), "Is the encoded label")

    if idx % 100 == 0:
        print ("Saving dictionary at loop: {}".format(idx))
        cPickle.dump(audio_embeddings_dict, open('audio_embeddings_dict_unbal_train_multi_{}.pkl'.format(idx), 'wb'))
        cPickle.dump(audio_multihot_dict, open('audio_multihot_dict_bal_untrain_multi_{}.pkl'.format(idx), 'wb'))
        cPickle.dump(audio_multihot_dict, open('audio_labels_unbal_dict_multi_{}.pkl'.format(idx), 'wb'))




pool = Pool(50)
result = pool.map(process_tfrecord, all_tfrecord_filenames)