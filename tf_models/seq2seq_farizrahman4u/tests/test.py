from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
#from keras.utils.test_utils import keras_test


max_encoder_length = 10
input_dim = 32

max_decoder_length = 8
output_dim = 150

batch = 100
hidden_dim = 80

np.random.seed(0)

#@keras_test
def test_SimpleSeq2Seq():
    x = np.random.random((batch, max_encoder_length, input_dim))
    y = np.random.random((batch, max_decoder_length, output_dim))

    models = []
    models += [SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=max_decoder_length, input_shape=(max_encoder_length, input_dim))]
    models += [SimpleSeq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=max_decoder_length, input_shape=(max_encoder_length, input_dim), depth=2)]

    for model in models:
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, nb_epoch=1)


#@keras_test
def test_Seq2Seq():
    x = np.random.random((batch, max_encoder_length, input_dim))
    y = np.random.random((batch, max_decoder_length, output_dim))

    models = []
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=max_decoder_length, input_shape=(max_encoder_length, input_dim))]
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=max_decoder_length, input_shape=(max_encoder_length, input_dim), peek=True)]
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=max_decoder_length, input_shape=(max_encoder_length, input_dim), depth=2)]
    models += [Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=max_decoder_length, input_shape=(max_encoder_length, input_dim), peek=True, depth=2)]

    for model in models:
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1)

    model = Seq2Seq(output_dim=output_dim, hidden_dim=hidden_dim, output_length=max_decoder_length, input_shape=(max_encoder_length, input_dim), peek=True, depth=2, teacher_force=True)
    model.compile(loss='mse', optimizer='sgd')
    model.fit([x, y], y, epochs=1)
    
#@keras_test
def test_AttentionSeq2Seq():
    print("test seq2seq-attention")
    x = np.random.random((batch, max_encoder_length, input_dim))
    y = np.random.random((batch, max_decoder_length, output_dim))

    models = []
    models += [AttentionSeq2Seq(output_dim=output_dim,
                                hidden_dim=hidden_dim,
                                output_length=max_decoder_length,
                                input_shape=(max_encoder_length, input_dim))]
    models += [AttentionSeq2Seq(output_dim=output_dim,
                                hidden_dim=hidden_dim,
                                output_length=max_decoder_length,
                                input_shape=(max_encoder_length, input_dim),
                                depth=2)]
    models += [AttentionSeq2Seq(output_dim=output_dim,
                                hidden_dim=hidden_dim,
                                output_length=max_decoder_length,
                                input_shape=(max_encoder_length, input_dim),
                                depth=3)]

    for model in models:
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1)

test_AttentionSeq2Seq()
