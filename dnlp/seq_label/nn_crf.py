# -*- coding: utf-8 -*-
"""Builds network of NNCRF, including training and inference,"""
import math
import numpy as np
import tensorflow as tf
from dnlp.config.seq_label_config import NeuralNetworkCRFConfig
from dnlp.utils.constant import MODE_FIT, MODE_INFERENCE, LOSS_LOG_LIKEHOOD, LOSS_MAX_MARGIN, NNCRF_DROPOUT_EMBEDDING, \
    NNCRF_DROPOUT_HIDDEN
from .nn_crf_base import NeuralNetworkCRFBase


class NeuralNetworkCRF(NeuralNetworkCRFBase):
    def __init__(self, mode, config: NeuralNetworkCRFConfig, label2result):
        super().__init__(mode=mode, config=config)
        self.label2result = label2result
        self.params = {}
        self.build_network()

    def build_network(self):
        """
        build the total network framework of NN-CRF, including initialize placeholders and variables,
        create network architecture,
        :return:
        """
        with self.graph.as_default():
            self.init_placeholder()
            self.embedding_layer = self.get_embedding_layer()
            if self.mode == MODE_FIT and self.config.dropout_position == NNCRF_DROPOUT_EMBEDDING:
                self.embedding_layer = self.get_dropout_layer(self.embedding_layer)
            self.hidden_layer = self.get_neural_network_layer(self.embedding_layer)
            if self.mode == MODE_FIT and self.config.dropout_position == NNCRF_DROPOUT_HIDDEN:
                self.hidden_layer = self.get_dropout_layer(self.hidden_layer)
            self.output = self.get_full_connected_layer(self.hidden_layer)
            self.init_crf_variable()
            if self.mode == MODE_FIT:
                regularizer = tf.contrib.layers.l2_regularizer(self.config.regularization_rate)
                self.regularization = tf.contrib.layers.apply_regularization(regularizer, tf.trainable_variables())
                if self.config.loss_function_name == LOSS_LOG_LIKEHOOD:
                    self.loss = self.get_log_likehood_loss()
                elif self.config.loss_function_name == LOSS_MAX_MARGIN:
                    self.loss = self.get_max_margin_loss()
                else:
                    raise Exception('loss function is not supported.')
                self.optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
                # self.optimizer = tf.train.AdagradOptimizer(self.config.learning_rate)
                self.train_model = self.optimizer.minimize(self.loss + self.regularization)

            self.sess.run(tf.global_variables_initializer())
            if self.mode == MODE_INFERENCE:
                tf.train.Saver().restore(save_path=self.config.model_name, sess=self.sess)

    def fit(self):
        print('start traininig......')
        with self.sess as sess:
            tf.global_variables_initializer().run()
            if self.config.loss_function_name == LOSS_LOG_LIKEHOOD:
                self.fit_log_likehood(sess)
            elif self.config.loss_function_name == LOSS_MAX_MARGIN:
                self.fit_max_margin(sess)
            else:
                raise Exception('loss function is not supported.')

    def inference(self, data, return_labels=False):
        if type(data) not in {str, list}:
            raise Exception('Input data type error, not string or list')
        elif isinstance(data, list):
            for s in data:
                if not isinstance(s, str):
                    raise Exception('list item is not string')
        elif isinstance(data, str):
            data = [data]

        sent_lengths = [len(sent) for sent in data]
        input_indices = self.sentences2input_indices(data, max(sent_lengths))
        runner = [self.output, self.transition, self.init_transition]
        feed_dict = {self.input_word_ids: input_indices, self.seq_length: sent_lengths}
        output, transition, init_transition = self.sess.run(runner, feed_dict=feed_dict)

        results = []
        for sent, sent_output in zip(data, output):
            labels = self.viterbi(sent_output.T, transition, init_transition)
            if not return_labels:
                results.append(self.label2result(sent, labels, self.label_schema))
            else:
                results.append(labels)

        return results

    def fit_log_likehood(self, sess, interval=1):
        saver = tf.train.Saver(max_to_keep=100)
        training_data = self.data.mini_batch(self.config.batch_size)
        for index, (words, labels, seq_lengths) in enumerate(training_data):
            if index % self.batch_count == 0:
                epoch = index // self.batch_count
                print('epoch {0}'.format(epoch))
                if epoch > 0 and epoch % interval == 0:
                    saver.save(sess, self.config.model_name.format(epoch))

            feed_dict = {self.input_word_ids: words, self.true_labels: labels, self.seq_length: seq_lengths}
            self.sess.run(self.train_model, feed_dict=feed_dict)
            print(np.sum(sess.run(self.loss, feed_dict=feed_dict)))

    def get_log_likehood_loss(self):
        with tf.name_scope('log_likehood'):
            crf_loss, _ = tf.contrib.crf.crf_log_likelihood(self.output,
                                                            self.true_labels,
                                                            self.seq_length,
                                                            self.transition)
            return -crf_loss / self.config.batch_size

    def fit_max_margin(self, sess, interval=1):
        saver = tf.train.Saver(max_to_keep=100)
        training_data = self.data.mini_batch(self.config.batch_size)
        for index, (words, labels, seq_lengths) in enumerate(training_data):
            if index % self.batch_count == 0:
                epoch = index // self.batch_count
                print('epoch {0}'.format(epoch))
                if epoch > 0 and epoch % interval == 0:
                    saver.save(sess, self.config.model_name.format(epoch))
            transition = self.transition.eval(session=sess)
            init_transition = self.init_transition.eval(session=sess)
            feed_dict = {self.input_word_ids: words, self.seq_length: seq_lengths}
            output = sess.run(self.output, feed_dict=feed_dict)
            pred_seq = []

            for i in range(self.config.batch_size):
                state = output[i, :seq_lengths[i], :].T
                seq = self._viterbi_training_stage(state, transition, init_transition, labels[i],
                                                   self.config.batch_length)
                pred_seq.append(seq)

            feed_dict = {self.true_seq: labels, self.pred_seq: pred_seq,
                         self.input_word_ids: words, self.seq_length: seq_lengths}
            sess.run(self.train_model, feed_dict=feed_dict)
            transition_diff = sess.run(self.transition_difference, feed_dict=feed_dict)
            state_diff = sess.run(self.state_difference, feed_dict=feed_dict)
            print('======================')
            print('trainsition: ', np.sum(transition_diff) / self.config.batch_size)
            print('state difference: ', np.sum(state_diff) / self.config.batch_size)
            print('loss: ', sess.run(self.loss, feed_dict=feed_dict))

    def get_max_margin_loss(self):
        with tf.name_scope('max_margin'):
            batch_index = np.repeat(np.expand_dims(np.arange(0, self.config.batch_size), 1),
                                    self.config.batch_length, 1)
            sent_index = np.repeat(np.expand_dims(np.arange(0, self.config.batch_length), 0),
                                   self.config.batch_size, 0)
            self.true_index = tf.stack([batch_index, sent_index, self.true_seq], axis=2)
            self.pred_index = tf.stack([batch_index, sent_index, self.pred_seq], axis=2)
            pred_state = tf.gather_nd(self.output, self.pred_index)
            true_state = tf.gather_nd(self.output, self.true_index)

            self.state_difference = tf.reduce_sum(pred_state - true_state, axis=1)
            pred_transition = tf.gather_nd(self.transition, tf.stack([self.pred_seq[:, :-1], self.pred_seq[:, 1:]], 2))
            true_transition = tf.gather_nd(self.transition, tf.stack([self.true_seq[:, :-1], self.true_seq[:, 1:]], 2))
            transition_mask = tf.sequence_mask(self.seq_length - 1, self.config.batch_length - 1, dtype=tf.float32)
            pred_transition = transition_mask * pred_transition
            true_transition = transition_mask * true_transition
            self.transition_difference = tf.reduce_sum(pred_transition - true_transition, axis=1)
            pred_init_transition = tf.gather_nd(self.init_transition, tf.expand_dims(self.pred_seq[:, 0], 1))
            true_init_transition = tf.gather_nd(self.init_transition, tf.expand_dims(self.true_seq[:, 0], 1))
            self.init_transition_difference = pred_init_transition - true_init_transition
            score_diff = self.state_difference + self.transition_difference + self.init_transition_difference
            hinge_loss = self.config.hinge_rate * tf.count_nonzero(self.pred_seq - self.true_seq, axis=1,
                                                                   dtype=tf.float32)
            # loss = tf.reduce_sum(tf.maximum(score_diff, -hinge_loss)) / self.config.batch_size
            loss = tf.reduce_sum(score_diff) / self.config.batch_size
            return loss

    def get_cross_entropy_softmax_loss(self):
        with tf.name_scope('cross_entropy_softmax'):
            pass

    def init_placeholder(self):
        self.input_word_ids = tf.placeholder(tf.int32, [None, None, None])
        self.true_labels = tf.placeholder(tf.int32, [None, None])
        self.seq_length = tf.placeholder(tf.int32, [None])
        if self.mode == MODE_FIT:
            self.true_seq = tf.placeholder(tf.int32, [self.config.batch_size, self.config.batch_length])
            self.pred_seq = tf.placeholder(tf.int32, [self.config.batch_size, self.config.batch_length])
            output_shape = [self.config.batch_size, self.config.batch_length, self.config.tag_count]
            self.output_placeholder = tf.placeholder(tf.float32, output_shape)

    def init_crf_variable(self):
        with tf.variable_scope('crf'):
            self.transition = tf.Variable(tf.random_uniform([self.label_count, self.label_count], -0.001, 0.001),
                                          name='transition')
            if self.config.loss_function_name == 'log likehood':
                self.init_transition = tf.Variable(tf.zeros([self.label_count]), name='init_transition',
                                                   trainable=False)
            elif self.config.loss_function_name == LOSS_MAX_MARGIN:
                self.init_transition = tf.Variable(tf.random_uniform([self.label_count], -0.001, 0.001),
                                                   name='init_transition')

    def get_embedding_layer(self):
        with tf.variable_scope('embedding'):
            embeddings = self.__get_variable([self.dict_size, self.config.word_embed_size], 'embeddings')
            self.params['embedding'] = embeddings
            if self.mode == MODE_FIT:
                input_size = [self.config.batch_size, self.config.batch_length, self.config.concat_embed_size]
                layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input_word_ids), input_size)
            else:
                layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input_word_ids),
                                   [1, -1, self.config.concat_embed_size])
            return layer

    def get_neural_network_layer(self, embedding_layer):
        with tf.variable_scope('neural_network'):
            hidden_layers = self.config.hidden_layers
            hidden_layer = self.__get_layer_by_type(hidden_layers[0]['type'], embedding_layer,
                                                    hidden_units=hidden_layers[0]['units'])
            for layer in hidden_layers[1:]:
                hidden_layer = self.__get_layer_by_type(layer['type'], hidden_layer, hidden_units=layer['units'])
            return hidden_layer

    def __get_layer_by_type(self, type_name, layer, **kwargs):
        if 'hidden_units' not in kwargs:
            raise Exception('don\'t assign hidden units')

        hidden_units = kwargs['hidden_units']

        if type_name == 'mlp':
            return self.get_mlp_layer(layer, hidden_units)
        elif type_name == 'rnn':
            return self.get_rnn_layer(layer, hidden_units)
        elif type_name == 'lstm':
            return self.get_lstm_layer(layer, hidden_units)
        elif type_name == 'gru':
            return self.get_gru_layer(layer, hidden_units)
        elif type_name == 'bidirectional_lstm':
            return self.get_bidirectional_lstm_layer(layer, hidden_units)
        elif type_name == 'bidirectional_gru':
            return self.get_bidirectional_gru_layer(layer, hidden_units)

    def get_mlp_layer(self, layer, hidden_units, name='mlp', weight_name='hidden_weight', bias_name='hidden_bias'):
        hidden_weight = self.__get_variable([hidden_units, self.config.concat_embed_size], name=weight_name)
        hidden_bias = tf.Variable(tf.random_uniform([hidden_units, 1, 1], -0.01, 0.01), name=bias_name)
        self.params['hidden_weight'] = hidden_weight
        self.params['hidden_bias'] = hidden_bias
        layer = tf.sigmoid(tf.tensordot(hidden_weight, tf.transpose(layer), [[1], [0]]) + hidden_bias, name=name)
        return tf.transpose(layer)

    def get_rnn_layer(self, layer, hidden_units, name='rnn'):
        rnn = tf.nn.rnn_cell.RNNCell(hidden_units)
        rnn_output, rnn_out_state = tf.nn.dynamic_rnn(rnn, layer, dtype=tf.float32)
        return rnn_output

    def get_lstm_layer(self, layer, hidden_units, name='lstm'):
        lstm = tf.nn.rnn_cell.LSTMCell(hidden_units)
        lstm_output, lstm_out_state = tf.nn.dynamic_rnn(lstm, layer, sequence_length=self.seq_length, dtype=tf.float32)
        return lstm_output

    def get_bidirectional_lstm_layer(self, layer, hidden_units, name='bidirectional_lstm'):
        lstm_fw = tf.nn.rnn_cell.LSTMCell(hidden_units // 2)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(hidden_units // 2)
        bilstm_output, bilstm_output_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, layer, self.seq_length,
                                                                             dtype=tf.float32)
        return tf.concat([bilstm_output[0], bilstm_output[1]], -1)

    def get_gru_layer(self, layer, hidden_units, name='gru'):
        gru = tf.nn.rnn_cell.GRUCell(hidden_units)
        gru_output, gru_out_state = tf.nn.dynamic_rnn(gru, layer, dtype=tf.float32)
        return gru_output

    def get_bidirectional_gru_layer(self, layer, hidden_units, name='bidirectional_gru'):
        gru_fw = tf.nn.rnn_cell.GRUCell(hidden_units // 2)
        gru_bw = tf.nn.rnn_cell.GRUCell(hidden_units // 2)
        gru_output, gru_output_state = tf.nn.bidirectional_dynamic_rnn(gru_fw, gru_bw, layer, self.seq_length,
                                                                       dtype=tf.float32)
        return tf.concat([gru_output[0], gru_output[1]], -1)

    def get_full_connected_layer(self, layer):
        hidden_units = self.config.hidden_layers[-1]['units']
        output_weight = self.__get_variable([hidden_units, self.config.tag_count], name='output_weight')
        output_bias = tf.Variable(tf.zeros([1, 1, self.config.tag_count]), name='output_bias')
        self.params['output_weight'] = output_weight
        self.params['output_bias'] = output_bias
        return tf.tensordot(layer, output_weight, [[2], [0]]) + output_bias

    def get_dropout_layer(self, layer):
        return tf.layers.dropout(layer, self.config.dropout_rate)

    def __get_variable(self, size, name):
        if name == 'embedding':
            return tf.Variable(tf.random_uniform(size, -0.01, -0.01), name=name)
        else:
            return tf.Variable(tf.truncated_normal(size, stddev=1.0 / math.sqrt(size[-1])), name=name)
