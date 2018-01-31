from multiprocessing import Process, Pipe
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Embedding
from keras.optimizers import Adam
import tensorflow as tf


def make_model(vecs, weights=None):
    inp = Input((5,))
    embd = Embedding(len(vecs), 50, weights=[vecs], trainable=False)(inp)
    out = Dense(5, activation='softmax')(embd)
    model = Model(inp, out)
    model.compile(Adam(0.001), 'categorical_crossentropy', metrics=['accuracy'])
    return model


def f(vecs, conn):
    model = make_model(vecs)
    conn.send('done')
    conn.close()


if __name__ == '__main__':
    vecs = np.random.random((100000, 50))
    model1 = make_model(vecs)

    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(vecs, child_conn), daemon=True)
    p.start()

    print('starting model two')
    print(parent_conn.recv())
    print('completed')