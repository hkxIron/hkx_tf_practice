import tensorflow as tf
import numpy as np

print("tf version: ", tf.__version__)

def t1():
    print("-"*50)
    num_examples = 10
    num_words = 5
    # All sequences in this example have the same length, but they can be variable in a real model.
    # 填充数组
    # Return a new array of given shape and type, filled with `fill_value`.
    sequence_lengths = np.full(shape=num_examples, fill_value=num_words - 1, dtype=np.int32)
    print(sequence_lengths)

t1()