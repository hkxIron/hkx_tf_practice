#!/usr/bin/env python
# coding=gbk
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige
#          \date   2016-08-12 11:52:01.952044
#   \Description   gen records for libsvm/tlc format dataset file
#                  this is just a demo code, reading one file and gen one file of TFRecord
#                  you may try to use multiprocess to do it parallel and gen multiple tf-records
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('label_type', 'int', '')

import numpy as np

_float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
_int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))


def main(argv):
    input = "data/input.txt"
    output = "data/output"
    writer = tf.python_io.TFRecordWriter(output)
    num = 0
    for line in open(input):
        if num % 10000 == 0:
            print('%d lines done' % num)
        l = line.rstrip().split()

        label = int(l[0]) if FLAGS.label_type == 'int' else float(l[0])

        # input can be libsmv or tlc format, for tlc format it contatins one col of num_features here will ignore
        start = 1
        if ':' not in l[1]:
            start += 1

        indexes = []
        values = []

        # notice most libsvm will use 1 as start index, and for tlc 0 as start index
        # so for libsvm format will assume 0 as non usesd
        for item in l[start:]:
            index, value = item.split(':')
            indexes.append(int(index))
            values.append(float(value))

        label_ = _int_feature([label]) if FLAGS.label_type == 'int' else _float_feature([label])
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': label_,
            'index': _int_feature(indexes),
            'value': _float_feature(values)
        }))
        writer.write(example.SerializeToString())
        num += 1


if __name__ == '__main__':
    tf.app.run()
