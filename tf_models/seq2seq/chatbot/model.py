import tensorflow as tf

class Seq2SeqModel():
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, mode, use_attention,
                 use_beam_search, beam_size, max_gradient_norm=5.0):
        self.learing_rate = learning_rate
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        self.mode = mode
        self.use_attention = use_attention
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm
        #执行模型构建部分的代码
        self.build_model()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            #添加dropout
            # output_keep_prob:  unit Tensor or float between 0 and 1, output keep probability; if it is constant and 1, no output dropout will be added.
            # 在rnn中进行dropout时，对于rnn的部分不进行dropout，也就是说从t-1时候的状态传递到t时刻进行计算时，这个中间不进行memory的dropout；仅在同一个t时刻中，多层cell之间传递信息的时候进行dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell

        #列表中每个元素都是调用single_rnn_cell函数
        multi_cells = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return multi_cells

    def build_model(self):
        print('building model... ...')
        #=================================1, 定义模型的placeholder
        # [batch, encoder_sequence_length]
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        # [batch]
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        # batch_size现在为变量,每次可以不同?
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        # decoder_targets:[batch, decoder_sequence_length]
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        # [batch]
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        # scalar,batch里最长的 decoder的序列长度
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.decoder_target_length_mask = tf.sequence_mask(lengths=self.decoder_targets_length, maxlen=self.max_target_sequence_length, dtype=tf.float32, name='masks')
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #=================================2, 定义模型的encoder部分
        with tf.variable_scope('encoder'):
            #创建LSTMCell，两层+dropout
            encoder_cell = self._create_rnn_cell()
            #构建embedding矩阵,encoder和decoder公用该词向量矩阵
            # embedding:[vocab_size, embedding_size]
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            # encoder_inputs:[batch, encoder_sequence_length]
            # encoder_inputs_embedded:[batch, encoder_sequence_length, embedding_size]
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)

            # 使用dynamic_rnn构建LSTM模型，将输入编码成隐层向量。
            # encoder_outputs用于attention，[batch_size,encoder_inputs_length,rnn_hidden_size]
            # encoder_state用于decoder的初始化状态
            # encoder_state:[hidden=[batch, rnn_hidden_size], cell=[batch, rnn_hidden_size]]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                               inputs=encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)

        # =================================3, 定义模型的decoder部分
        with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            # if self.beam_search:
            #     # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
            #     print("use beamsearch decoding..")
            #     encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
            #     encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
            #     encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

            """
            Bahdanau注意力机制，还有一种注意力机制称为Luong注意力机制，二者最主要的区别是前者为加法注意力机制，后者为乘法注意力机制
            详见: https://github.com/tensorflow/nmt
            bahdanau:
            score(h_t,h_s)= v_a^T*tanh(W1*h_t+W2*h_s)
            
            luong:
            score(h_t,h_s)= h_t^T*W*h_s
            """
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size,
                                                                       memory=encoder_outputs,
                                                                       memory_sequence_length=encoder_inputs_length)
            #attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要是用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = self._create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size,
                                                               name='Attention_Wrapper')
            #如果使用beam_seach则batch_size = self.batch_size * self.beam_size。因为之前已经复制过一次
            #batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size
            batch_size = self.batch_size

            #定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            # output_layer:[batch, decoder_sequence_length, vocab_size]
            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            if self.mode == 'train':
                # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
                # 即将 A,B,C,<end> => <go>,A,B,C
                # decoder_inputs_embedded的shape为[batch_size, decoder_targets_length, embedding_size]

                # decoder_targets: [batch, decoder_sequence_length]
                ending = tf.strided_slice(input_=self.decoder_targets, begin=[0, 0], end=[self.batch_size, -1], strides=[1, 1])
                # decoder_input: [batch, decoder_sequence_length]
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<go>']), ending], 1)
                # decoder_input_embedded: [batch, decoder_sequence_length, embedding_size]
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
                #训练阶段，使用TrainingHelper+BasicDecoder的组合，这一般是固定的，当然也可以自己定义Helper类，实现自己的功能
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False,
                                                                    name='training_helper')
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                   helper=training_helper,
                                                                   initial_state=decoder_initial_state, # decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
                                                                   output_layer=output_layer) # [batch, decoder_sequence_length, vocab_size]
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size, decoder_targets_length], tf.int32，保存最终的编码结果。可以表示最后的答案
                # 数学公式:
                # ht = g(h(t-1), y(t-1), c(t))
                # yt = softmax(ht, y(t-1), c(t)), softmax(decoder_outputs) => yt
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                          impute_finished=True, # 是否有mask
                                                                          maximum_iterations=self.max_target_sequence_length)
                # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                # decoder_predict_train: [batch_size, decoder_targets_length]
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                # 使用sequence_loss计算loss，这里需要传入之前定义的mask标志
                # decoder_logits_train:[batch_size, decoder_targets_length, vocab_size]
                # decoder_targets:[batch_size, decoder_targets_length]
                # decoder_target_length_mask:[batch, max_target_sequence_length]
                # loss: scalar
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets,
                                                             weights=self.decoder_target_length_mask)

                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

                """
                
                if clip > 0: # gradient clipping if clip is positive
                    grads, vs     = zip(*optimizer.compute_gradients(loss))
                    grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                    self.train_op = optimizer.apply_gradients(zip(grads, vs), global_step = global_step)
                else:
                    self.train_op = optimizer.minimize(loss, global_step = global_step)
                """

            elif self.mode == 'decode':
                # start_tokens:[batch, ]
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<go>']
                end_token = self.word_to_idx['<eos>']
                # decoder阶段根据是否使用beam_search决定不同的组合，
                # 如果使用则直接调用BeamSearchDecoder（里面已经实现了helper类）
                # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
                if self.use_beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                                             embedding=embedding,
                                                                             start_tokens=start_tokens,
                                                                             end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=output_layer)
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens,
                                                                               end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=decoding_helper, # helper主要用来处理embedding
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=output_layer)

                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                maximum_iterations=10)
                # 调用dynamic_decode进行解码，decoder_outputs是一个namedtuple，
                # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32

                # 对于使用beam_search的时候，它里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size],保存输出结果
                # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # 所以对应只需要返回predicted_ids或者sample_id即可翻译成最终的结果
                if self.use_beam_search:
                    # predicted_ids: [batch_size, decoder_targets_length, beam_size]
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    # predicted_ids: [batch_size, decoder_targets_length, 1]
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
        # =================================4, 保存模型
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = { self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 0.5,
                      self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary, global_step = sess.run([self.train_op, self.loss, self.summary_op, self.global_step], feed_dict=feed_dict)
        return loss, summary, global_step

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = { self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        #infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = { self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict