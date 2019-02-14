import os
import numpy as np
import tensorflow as tf
from vocabulary import Vocabulary


class Seq2SeqModel(object):
    """
        Seq2Seq model class
        """

    def __init__(self, mode, model_hparams, vocabulary, model_dir):
        """
             create Seq2Seq model

            Args:
                mode: 'train' or 'infer'
                model_hparams: hyper-parameters of the model
                vocabulary: word embedding matrix of input and output
                model_dir: directory to output model summaries and checkpoint
            """
        self.mode = mode
        self.model_hparams = model_hparams
        self.vocabulary = vocabulary
        self. model_dir = model_dir

        if self.mode == 'train' or self.model_hparams.beam_width is None:
            self.beam_width = 0
        else:
            self.beam_width = self.model_hparams.beam_width

        # reset the default TF graph
        tf.reset_default_graph()

        # define model inputs
        self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.float32, name='input_batch')
        self.input_batch_lengths = tf.placeholder(shape=(None, ), dtype=tf.int32, name='input_batch_lengths')

        # build model
        initializer_feed_dict = {}

        # train mode
        if self.mode == 'train':
            # define training model inputs
            self.target_batch = tf.placeholder(shape=(None, None), dtype=tf.float32, name='target_batch')
            self.target_batch_lengths = tf.placeholder(shape=(None, None), dtype=tf.float32,
                                                       name='target_batch_lengths')
            self. dropout_rate = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])
            if self.vocabulary.external_embeddings_matrix is not None:
                self.embeddings_matrix = tf.placeholder(shape=self.vocabulary.external_embeddings_matrix.shape, dtype=tf.float32,
                                                        name='external_embeddings')
                initializer_feed_dict[self.embeddings_matrix] = self.vocabulary.external_embeddings_matrix

            self.loss, self.training_step = self._build_model()

        elif self.mode == 'infer':
            # define inference model inputs
            self.max_output_sequence_lengths = tf.placeholder(shape=[], dtype=tf.float32,
                                                              name='max_output_sequence_lengths')
            self.beam_length_penalty_weight = tf.placeholder(dtype=tf.float32, name='beam_length_penalty_weight')

            self.predictions, self.predictions_seq_lengths = self._build_model()

        else:
            raise ValueError("mode must be 'train' or 'infer'.")

        # define new session and checkpoint saver
        self.session = self._create_session()
        self.session.run(tf.global_variables_initializer(), initializer_feed_dict)
        self.saver = tf.train.Saver()

    def load(self, filename):
        """
            load a trained model from a checkpoint

            Args:
                filename: checkpoint filename, such as model_checkpoint.ckpy
            """
        filepath = os.path.join(self.model_dir, filename)
        self.saver.restore(self.session, filepath)

    def save(self, filename):
        """
            Args:
                filename: checkpoint filename, such as model_checkpoint.ckpt
            """
        filepath = os.path.join(self.model_dir, filename)
        self.saver.save(self.session, filepath)

    def train_batch(self, inputs, targets, input_batch_lengths, target_batch_lengths,
                    learning_rate, dropout_rate):
        """
        :param inputs:  input matrix of shape (batch_size, batch_length)
        :param targets:  target matrix of shape (batch_size, batch_length)
        :param input_batch_lengths:  vector of sequence lengths of shape (batch_size)
        :param target_batch_lengths:  vector of sequence lengths of shape (batch_size)
        :param learning_rate:  learning rate for training
        :param dropout_rate:  dropout rate for training
        # :param global_step: index of training step across all batches and all epochs

        :return: batch_training_loss
        """
        if self.mode != 'train':
            raise ValueError("train_batch can only be called in train mode")

        # train on batch
        _, batch_training_loss = self.session.run([self.training_step, self.loss],
                                                  {self.input_batch: inputs,
                                                   self.target_batch: targets,
                                                   self.input_batch_lengths: input_batch_lengths,
                                                   self.target_batch_lengths: target_batch_lengths,
                                                   self.learning_rate: learning_rate,
                                                   self.dropout_rate: dropout_rate})

        return batch_training_loss

    def validate_batch(self, inputs, targets, input_batch_lengths, target_batch_lengths):
        """
        :param inputs: input matrix of shape (batch_size, batch_length)
        :param targets: target matrix of shape (batch_size, batch_length)
        :param input_batch_lengths: vector of sequence lengths of shape (batch_size)
        :param target_batch_lengths: vector of sequence lengths of shape (batch_size)

        :return: metric_value
        """
        if self.mode != 'train':
            raise ValueError("validate_batch can only be called in train mode")

        metric_op = self.loss
        metric_value = self.session.run(metric_op, {self.input_batch: inputs,
                                                    self.target_batch: targets,
                                                    self.input_batch_lengths: input_batch_lengths,
                                                    self.target_batch_lengths: target_batch_lengths,
                                                    self.dropout_rate: 0})

        return metric_value

    def predict_batch(self, inputs, input_batch_lengths, max_output_sequence_length, beam_length_penalty_weight):
        """
        :param inputs: input matrix of shape (batch_size, batch_length)
        :param input_batch_lengths: vector of sequence lengths of shape (batch_size)
        :param max_output_sequence_length: max number of timesteps the decoder can generate
        :param beam_length_penalty_weight: influences how beams are ranked.
                                        large negative values => very short beams first
                                        large positive values => very long beams first
                                        positive value between 0 and 2 should be enough for chatbot

        :return: predicted_output_info[0]
        """
        if self.mode != 'infer':
            raise ValueError("predict_batch can only be called in infer mode")

        fetches = [{'predictions': self.predictions,
                    'predictions_seq_lengths': self.predictions_seq_lengths}]

        predicted_output_info = self.session.run(fetches, {self.input_batch: inputs,
                                                           self.input_batch_lengths: input_batch_lengths,
                                                           self.max_output_sequence_lengths: max_output_sequence_length,
                                                           self.beam_length_penalty_weight: beam_length_penalty_weight})

        return predicted_output_info[0]

    def _build_model(self):
        """
            create Seq2Seq model graph
        """
        with tf.variable_scope('model'):
            batch_size = tf.shape(self.input_batch)[0]

            # encoder
            with tf.variable_scope('encoder'):
                encoder_embeddings_matrix_shape = [self.vocabulary.size(), self.model_hparams.embeddings_size]
                encoder_embeddings_matrix_initial_value = self.embeddings_matrix
                encoder_embedding_matrix = tf.Variable(encoder_embeddings_matrix_initial_value,
                                                       name='embeddings_matrix',
                                                       trainable=self.model_hparams.embedding_trainable,
                                                       expected_shape=encoder_embeddings_matrix_shape)

                encoder_embedded_input = tf.nn.embedding_lookup(encoder_embedding_matrix, self.input_batch)

                encoder_outputs, encoder_state = self._build_encoder(encoder_embedded_input)

            # decoder
            with tf.variable_scope('decoder') as decoder_scope:
                decoder_embeddings_matrix = encoder_embedding_matrix

                # create attentional decoder cell
                decoder_cell, decoder_initial_state = self._build_attention_decoder_cell(encoder_outputs,
                                                                                         encoder_state,
                                                                                         batch_size)
                # output layer
                weights = tf.truncated_normal_initializer(stddev=0.1)
                # biases = tf.zeros_initializer()
                output_layer = tf.layers.dense(units=self.vocabulary.size(), kernal_initializer=weights, use_bias=True)

                # build decoder RNN using attentional decoder cell and output layer
                if self.mode != 'infer':
                    loss, training_step = self._build_training_decoder(batch_size,
                                                                       decoder_embeddings_matrix,
                                                                       decoder_cell,
                                                                       decoder_initial_state,
                                                                       decoder_scope,
                                                                       output_layer)

                    return loss, training_step
                else:
                    predictions, predictions_seq_lengths = self._build_inference_decoder(batch_size,
                                                                                         decoder_embeddings_matrix,
                                                                                         decoder_cell,
                                                                                         decoder_initial_state,
                                                                                         decoder_scope,
                                                                                         output_layer)

                    return predictions, predictions_seq_lengths

    def _build_encoder(self, encoder_embedded_input):
        """
            create encoder RNN
        :param encoder_embedded_input: embedded input sequences

        :return: encoder_outputs, encoder_state
        """
        keep_prob = 1 - self.dropout_rate if self.mode == 'train' else None
        if self.model_hparams.use_bidirectional_encoder:
            num_bi_layers = int(self.model_hparams.encoder_num_layers / 2)

            encoder_cell_forward = self._create_rnn_cell(self.model_hparams.rnn_size, num_bi_layers, keep_prob)
            encoder_cell_backward = self._create_rnn_cell(self.model_hparams.rnn_size, num_bi_layers, keep_prob)

            bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_forward,
                                                                                   cell_bw=encoder_cell_backward,
                                                                                   sequence_length=self.input_batch_lengths,
                                                                                   inputs=encoder_embedded_input,
                                                                                   dtype=tf.float32,
                                                                                   swap_memory=True)

            # manipulating encoder state to handle multi bidirectional layers
            encoder_outputs = tf.concat(bi_encoder_outputs, -1)

            if num_bi_layers == 1:
                encoder_state = bi_encoder_state
            else:
                encoder_state = []
                for layer_id in range(num_bi_layers):
                    encoder_state.append(bi_encoder_state[0][layer_id])
                    encoder_state.append(bi_encoder_state[1][layer_id])
                encoder_state = tuple(encoder_state)

        else:
            # uni-directional RNN
            encoder_cell = self._create_rnn_cell(self.model_hparams.rnn_size, self.model_hparams.encoder_num_layers,
                                                 keep_prob)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                               sequence_length=self.input_batch_lengths,
                                                               inputs=encoder_embedded_input,
                                                               dtype=tf.float32,
                                                               swap_memory=True)

        return encoder_outputs, encoder_state

    def _build_attention_decoder_cell(self, encoder_outputs, encoder_state, batch_size):
        """
        :param encoder_outputs: a tensor containing the output of the encoder at each timestep
        :param encoder_state: a tensor containing the final encoder state for each encoder cell
        :param batch_size: batch size tensor

        :return: attention_decoder_cell, decoder_initial_state
        """
        # If beam search decoding - repeat the input sequence length, encoder output, encoder state,
        # and batch size tensors once for every beam.
        input_sequence_length = self.input_batch_lengths
        if self.beam_width > 0:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_width)
            input_sequence_length = tf.contrib.seq2seq.tile_batch(input_sequence_length, multiplier=self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.beam_width)
            batch_size = batch_size * self.beam_width

        # construct attention mechanism
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.model_hparams.rnn_size,
                                                                   memory=encoder_outputs,
                                                                   memory_sequence_length=input_sequence_length)

        # create decoder cell and wrap with attention mechanism
        with tf.variable_scope('decoder_cell'):
            keep_prob = 1 - self.dropout_rate if self.mode == 'train' else None
            decoder_cell = self._create_rnn_cell(self.model_hparams.rnn_size,
                                                 self.model_hparams.decoder_num_layers,
                                                 keep_prob)

            alignment_history = self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width == 0
            # output_attention = self.model_hparams.attention_type == 'luong' or self.model_hparams.attention_type == 'scaled_luong'
            attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell,
                                                                         attention_mechanism=attention_mechanism,
                                                                         attention_layer_size=self.model_hparams.rnn_size,
                                                                         alignment_history=alignment_history,
                                                                         name='attention_decoder_cell')

            # if encoder and decoder are the same structure, set the decoder initial state to the encoder final state
            decoder_initial_state = attention_decoder_cell.zero_state(batch_size, tf.float32)
            if self.model_hparams.encoder_num_layers == self.model_hparams.decoder_num_layers:
                decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

        return attention_decoder_cell, decoder_initial_state

    def _build_training_decoder(self, batch_size, decoder_embeddings_matrix, decoder_cell,
                                decoder_initial_state, decoder_scope, output_layer):
        """
        build decoder RNN for train mode

        :param batch_size: batch size tensor
        :param decoder_embeddings_matrix: decoder embeddings matrix
        :param decoder_cell: decoder RNN cell
        :param decoder_initial_state: initial cell state of decoder
        :param decoder_scope: scope of decoder
        :param output_layer:

        :return: loss, training_step
        """
        # preprocess each target sequence with the <SOS> token,
        # which is always the first input of first decoder timestep
        preprocessed_targets = self._preprocess_targets(batch_size)
        decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)

        # create training decoder
        helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embedded_input,
                                                   sequence_length=self.target_batch_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                  helper=helper,
                                                  initial_state=decoder_initial_state)

        # get the decoder output
        decoder_output, _final_context_state, _final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                              swap_memory=True,
                                              scope=decoder_scope)

        # pass the decoder output through the dense layer which will become a softmax distribution of classes -
        # one class per word in the output vocabulary
        logits = output_layer(decoder_output.rnn_output)

        # calculate softmax loss
        loss_mask = tf.sequence_mask(self.target_batch_lengths, tf.shape(self.target_batch)[1], dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=self.target_batch,
                                                weights=loss_mask)

        # set up optimizer
        if self.model_hparams.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.model_hparams.optimizer == 'adam':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise ValueError("use 'sgd' or 'adam")

        if self.model_hparams.max_gradient_norm > 0.0:
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.model_hparams.max_gradient_norm)
            training_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        else:
            training_step = optimizer.minimize(loss=loss)

        return loss, training_step

    def _build_inference_decoder(self, batch_size, decoder_embeddings_matrix, decoder_cell, decoder_initial_state,
                                 decoder_scope, output_layer):
        """Build the decoder RNN for inference mode.

        Args:
            See _build_training_decoder
        """
        # Get the SOS and EOS tokens
        start_tokens = tf.fill([batch_size], self.vocabulary.sos_int())
        end_token = self.vocabulary.eos_int()

        # Build the beam search, greedy, or sampling decoder
        if self.beam_width > 0:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                           embedding=decoder_embeddings_matrix,
                                                           start_tokens=start_tokens,
                                                           end_token=end_token,
                                                           initial_state=decoder_initial_state,
                                                           beam_width=self.beam_width,
                                                           output_layer=output_layer,
                                                           length_penalty_weight=self.beam_length_penalty_weight)
        else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=decoder_embeddings_matrix,
                                                              start_tokens=start_tokens,
                                                              end_token=end_token)

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                      helper=helper,
                                                      initial_state=decoder_initial_state,
                                                      output_layer=output_layer)

        # Get the decoder output
        decoder_output, final_context_state, final_sequence_lengths = \
            tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                              maximum_iterations=self.max_output_sequence_lengths,
                                              swap_memory=True,
                                              scope=decoder_scope)

        # Return the predicted sequences along with an array of the sequence lengths
        # for each predicted sequence in the batch
        if self.beam_width > 0:
            predictions = decoder_output.predicted_ids
            predictions_seq_lengths = final_context_state.lengths
        else:
            predictions = decoder_output.sample_id
            predictions_seq_lengths = final_sequence_lengths

        return predictions, predictions_seq_lengths

    def _create_rnn_cell(self, rnn_size, num_layers, keep_prob):
        """
        create a single RNN cell or stack of RNN cells

        :param rnn_size: number of neurons in each RNN cell
        :param num_layers: number of stacked RNN cells to create
        :param keep_prob: probability of not being dropped out; None for inference mode

        :return: single RNN cell or stack of RNN cells
        """
        cells = []
        for _ in range(num_layers):
            if self.model_hparams.rnn_cell_type == 'lstm':
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)
            elif self.model_hparams.rnn_cell_type == 'gru':
                rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_size)
            else:
                raise ValueError("use 'lstm' or 'gru'.")

            if keep_prob is not None:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell,
                                                         input_keep_prob=keep_prob)

            cells.append(rnn_cell)

        if len(cells) == 1:
            return cells[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cells=cells)

    def _preprocess_targets(self, batch_size):
        """
        prepend the SOS token to all target sequences in the batch

        :param batch_size: batch size tensor

        :return: preprocessed_target
        """
        left_side = tf.fill([batch_size, 1], self.vocabulary.sos_int())
        right_side = tf.strided_slice(self.target_batch, [0, 0], [batch_size, -1], [1, 1])
        preprocessed_targets = tf.concat([left_side, right_side], 1)

        return preprocessed_targets

    def _create_session(self):
        """
        initialize tensorflow session
        """
        session = tf.Session()

        return session

    def chat(self, question, chat_settings):
        """
        chat with the seq2seq model

        :param question: input question in which the model should predict an answer
        :param chat_settings: chat settings

        :return: answer
        """
        # Process the question by cleaning it and converting it to an integer encoded vector
        if chat_settings.enable_auto_punctuation:
            question = Vocabulary.auto_punctuate(question)
        question = Vocabulary.clean_text(question, normalize_words=chat_settings.inference_hparams.normalize_words)
        question = self.vocabulary.words2ints(question)

        # Get the answer prediction
        batch = np.expand_dims(question, 0)
        max_output_sequence_length = chat_settings.inference_hparams.max_answer_words + 1
        predicted_answer_info = self.predict_batch(inputs=batch,
                                                   input_batch_lengths=1,
                                                   max_output_sequence_length=max_output_sequence_length,
                                                   beam_length_penalty_weight=chat_settings.inference_hparams.beam_length_penalty_weight)

        # Read the answer prediction
        answer_beams = []
        if self.beam_width > 0:
            # For beam search decoding: if show_all_beams is enabeled then output all beams (sequences),
            # otherwise take the first beam.
            # The beams (in the "predictions" matrix) are ordered with the highest ranked beams first.
            beam_count = 1 if not chat_settings.show_all_beams else len(
                predicted_answer_info["predictions_seq_lengths"][0])
            for i in range(beam_count):
                predicted_answer_seq_length = predicted_answer_info["predictions_seq_lengths"][0][
                                                  i] - 1  # -1 to exclude the EOS token
                predicted_answer = predicted_answer_info["predictions"][0][:predicted_answer_seq_length, i].tolist()
                answer_beams.append(predicted_answer)
        else:
            # For greedy / sampling decoding: only one beam (sequence) is returned,
            # based on the argmax for greedy decoding
            # or the sampling distribution for sampling decoding. Return this beam.
            beam_count = 1
            predicted_answer_seq_length = predicted_answer_info["predictions_seq_lengths"][
                                              0] - 1  # -1 to exclude the EOS token
            predicted_answer = predicted_answer_info["predictions"][0][:predicted_answer_seq_length].tolist()
            answer_beams.append(predicted_answer)

        # Convert the answer(s) to text and return
        answers = []
        for i in range(beam_count):
            answer = self.vocabulary.ints2words(answer_beams[i])
            answers.append(answer)

        if chat_settings.show_all_beams:
            return answers
        else:
            return answers[0]
