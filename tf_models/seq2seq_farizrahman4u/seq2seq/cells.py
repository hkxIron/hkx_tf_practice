import recurrentshop
from recurrentshop.cells import *
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Activation
from keras.layers import add, multiply, concatenate
from keras import backend as K


class LSTMDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim
        super(LSTMDecoderCell, self).__init__(**kwargs)

    def build_model(self, input_shape):
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim

        x = Input(batch_shape=input_shape)
        h_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))
        c_tm1 = Input(batch_shape=(input_shape[0], hidden_dim))

        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,
                   use_bias=False)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer,)
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer,)

        z = add([W1(x), U(h_tm1)])

        z0, z1, z2, z3 = get_slices(z, 4)
        i = Activation(self.recurrent_activation)(z0)
        f = Activation(self.recurrent_activation)(z1)
        c = add([multiply([f, c_tm1]), multiply([i, Activation(self.activation)(z2)])])
        o = Activation(self.recurrent_activation)(z3)
        h = multiply([o, Activation(self.activation)(c)])
        y = Activation(self.activation)(W2(h))

        return Model([x, h_tm1, c_tm1], [y, h, c])


class AttentionDecoderCell(ExtendedRNNCell):

    def __init__(self, hidden_dim=None, **kwargs):
        if hidden_dim:
            self.hidden_dim = hidden_dim
        else:
            self.hidden_dim = self.output_dim
        self.input_ndim = 3
        super(AttentionDecoderCell, self).__init__(**kwargs)


    def build_model(self, input_shape):
        # input_shape:[None, input_length=2, input_dim=2] , 不过我不清楚2,2是如何来的
        input_dim = input_shape[-1] # 2
        output_dim = self.output_dim # output_dim = 150
        input_length = input_shape[1] # 2
        hidden_dim = self.hidden_dim # hidden_dim=80
        # x:[batch, input_length=2, input_dim=2]
        x = Input(batch_shape=input_shape)
        hidden_last_time = Input(batch_shape=(input_shape[0], hidden_dim)) # [batch, hidden]
        cell_last_time = Input(batch_shape=(input_shape[0], hidden_dim)) # [batch, hidden]
        # W1:
        W1 = Dense(hidden_dim * 4,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W2 = Dense(output_dim,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        W3 = Dense(1,
                   kernel_initializer=self.kernel_initializer,
                   kernel_regularizer=self.kernel_regularizer)
        U = Dense(hidden_dim * 4,
                  kernel_initializer=self.kernel_initializer,
                  kernel_regularizer=self.kernel_regularizer)
        # C: [batch, input_length=2, hidden]
        C = Lambda(lambda x: K.repeat(x, input_length),
                   output_shape=(input_length, input_dim))(cell_last_time)
        # _xC:[batch, input_length=2, input_dim+hidden = 82]
        _xC = concatenate([x, C], axis=-1)
        # _xC:[batch*input_length, input_dim+hidden]
        _xC = Lambda(lambda x: K.reshape(x, (-1, input_dim + hidden_dim)),
                     output_shape=(input_dim + hidden_dim,))(_xC)
        # alpha:[batch*input_length,1]
        alpha = W3(_xC)
        alpha = Lambda(lambda x: K.reshape(x, (-1, input_length)),
                       output_shape=(input_length,))(alpha) # alpha:[batch, input_length]
        alpha = Activation('softmax')(alpha) # alpha:[batch, input_length]
        # alpha:[batch, input_length], x:[batch, input_length=2, input_dim=2], _x:[batch, input_length=2]
        """
        alpha:[batch,input_length]->[batch,input_length,1]
        x:[batch, input_length=2, input_dim=2]
        out = tf.matmul(alpha, x, adjoint_a=True, adjoint_b=False)
        out:[batch, 1, input_dim=2]
        out:tf.squeeze, [batch, input_dim=2]
        """
        # _x:[batch, input_dim=2]
        _x = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 1)),
                    output_shape=(input_dim,))([alpha, x])
        W1x = W1(_x) # W1x:[batch, hidden_dim*4]
        # hidden_last_time:[batch, hidden], U:[hidden, hidden*4] => Uh_last_time:[batch, hidden_dim*4]
        Uh_last_time= U(hidden_last_time)
        z = add([W1x, Uh_last_time]) # w1*x+U*hidden_last_time, z: [batch, hidden_dim*4]
        """
        lstm:
        input_gate = sigmoid(W_ix*input_x + W_ih*h_(t-1) + bias_ix)
        forget_gate = sigmoid(W_fx*input_x + W_fh*h_(t-1) + bias_fx)
        output_gate = sigmoid(W_ox*input_x + W_oh*h_(t-1) + bias_ox)
        cell_candidate = tanh(W_cx*input_x + W_ch*h_(t-1) + bias_cx)
        c_t = forget_gate*c_(t-1) + input_gate*cell_candidate
        h_t = o_t*tanh(c_t)
        """

        # z0,z1,z2,z3:[batch, hidden],z0:
        z0, z1, z2, z3 = get_slices(z, 4)

        input_gate = Activation(self.recurrent_activation)(z0)  # input_gate:[batch, hidden]
        forget_gate = Activation(self.recurrent_activation)(z1) # forget_gate:[batch, hidden]
        output_gate = Activation(self.recurrent_activation)(z3) # output_gate:[batch, hidden]
        cell_last_candidate = Activation(self.activation)(z2) # cell_state_candidate:[batch, hidden]
        # cell_state: [batch, hidden]
        cell_state = add([
                        multiply([forget_gate, cell_last_time]),
                        multiply([input_gate, cell_last_candidate]),
                        ])
        # hidden: [batch, hidden]
        hidden = multiply([output_gate, Activation(self.activation)(cell_state)])
        output = Activation(self.activation)(W2(hidden)) # output:[batch, output_dim=150], output=g(w*h_(t))
        model = Model(inputs=[x, hidden_last_time, cell_last_time], outputs=[output, hidden, cell_state])
        return model
