# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math
from dnlp.core.dnn_crf_base import DnnCrfBase
from dnlp.config.config import DnnCrfConfig


class DnnCrf(DnnCrfBase):
  def __init__(self, *, config: DnnCrfConfig = None, data_path: str = '', dtype: type = tf.float32, mode: str = 'train',
               train: str = 'll', nn: str, model_path: str = ''):
    if mode not in ['train', 'predict']:
      raise Exception('mode error')
    if nn not in ['mlp', 'rnn', 'lstm', 'bilstm', 'gru']:
      raise Exception('name of neural network entered is not supported')

    DnnCrfBase.__init__(self, config, data_path, mode, model_path)
    self.dtype = dtype
    self.mode = mode

    # 构建
    tf.reset_default_graph()
    self.transition = self.__get_variable([self.tags_count, self.tags_count], 'transition')
    self.transition_init = self.__get_variable([self.tags_count], 'transition_init')
    self.params = [self.transition, self.transition_init]
    # 输入层
    if mode == 'train':
      self.input = tf.placeholder(tf.int32, [self.batch_size, self.batch_length, self.windows_size])
      self.real_indices = tf.placeholder(tf.int32, [self.batch_size, self.batch_length])
      self.seq_length = tf.placeholder(tf.int32, [self.batch_size])
    else:
      self.input = tf.placeholder(tf.int32, [None, self.windows_size])

    # 查找表层
    self.embedding_layer = self.get_embedding_layer()
    # 隐藏层
    if nn == 'mlp':
      self.hidden_layer = self.get_mlp_layer(tf.transpose(self.embedding_layer))
    elif nn == 'lstm':
      self.hidden_layer = self.get_lstm_layer(self.embedding_layer)
    elif nn == 'gru':
      self.hidden_layer = self.get_gru_layer(tf.transpose(self.embedding_layer))
    else:
      self.hidden_layer = self.get_rnn_layer(tf.transpose(self.embedding_layer))
    # 输出层
    self.output = self.get_output_layer(self.hidden_layer)

    if mode == 'predict':
      self.output = tf.squeeze(tf.transpose(self.output), axis=2)
      self.sess = tf.Session()
      self.sess.run(tf.global_variables_initializer())
      tf.train.Saver().restore(save_path=self.model_path, sess=self.sess)
    elif train == 'll':
      self.ll_loss, _ = tf.contrib.crf.crf_log_likelihood(self.output, self.real_indices, self.seq_length,
                                                          self.transition)
      self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
      self.train_ll = self.optimizer.minimize(-self.ll_loss)
    else:
      # 构建训练函数
      # 训练用placeholder
      self.ll_corr = tf.placeholder(tf.int32, shape=[None, 3])
      self.ll_curr = tf.placeholder(tf.int32, shape=[None, 3])
      self.trans_corr = tf.placeholder(tf.int32, [None, 2])
      self.trans_curr = tf.placeholder(tf.int32, [None, 2])
      self.trans_init_corr = tf.placeholder(tf.int32, [None, 1])
      self.trans_init_curr = tf.placeholder(tf.int32, [None, 1])
      # 损失函数
      self.loss, self.loss_with_init = self.get_loss()
      self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
      self.train = self.optimizer.minimize(self.loss)
      self.train_with_init = self.optimizer.minimize(self.loss_with_init)

  def fit(self, epochs: int = 100, interval: int = 20):
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(max_to_keep=100)
      for epoch in range(1, epochs + 1):
        print('epoch:', epoch)
        for _ in range(self.batch_count):
          characters, labels, lengths = self.get_batch()
          self.fit_batch(characters, labels, lengths, sess)
        # if epoch % interval == 0:
        model_path = '../dnlp/models/cws{0}.ckpt'.format(epoch)
        saver.save(sess, model_path)
        self.save_config(model_path)

  def fit_batch(self, characters, labels, lengths, sess):
    scores = sess.run(self.output, feed_dict={self.input: characters})
    transition = self.transition.eval(session=sess)
    transition_init = self.transition_init.eval(session=sess)
    update_labels_pos = None
    update_labels_neg = None
    current_labels = []
    trans_pos_indices = []
    trans_neg_indices = []
    trans_init_pos_indices = []
    trans_init_neg_indices = []
    for i in range(self.batch_size):
      current_label = self.viterbi(scores[:, :lengths[i], i], transition, transition_init)
      current_labels.append(current_label)
      diff_tag = np.subtract(labels[i, :lengths[i]], current_label)
      update_index = np.where(diff_tag != 0)[0]
      update_length = len(update_index)
      if update_length == 0:
        continue
      update_label_pos = np.stack([labels[i, update_index], update_index, i * np.ones([update_length])], axis=-1)
      update_label_neg = np.stack([current_label[update_index], update_index, i * np.ones([update_length])], axis=-1)
      if update_labels_pos is not None:
        np.concatenate((update_labels_pos, update_label_pos))
        np.concatenate((update_labels_neg, update_label_neg))
      else:
        update_labels_pos = update_label_pos
        update_labels_neg = update_label_neg

      trans_pos_index, trans_neg_index, trans_init_pos, trans_init_neg, update_init = self.generate_transition_update_index(
        labels[i, :lengths[i]], current_labels[i])

      trans_pos_indices.extend(trans_pos_index)
      trans_neg_indices.extend(trans_neg_index)

      if update_init:
        trans_init_pos_indices.append(trans_init_pos)
        trans_init_neg_indices.append(trans_init_neg)

    if update_labels_pos is not None and update_labels_neg is not None:
      feed_dict = {self.input: characters, self.ll_curr: update_labels_neg, self.ll_corr: update_labels_pos,
                   self.trans_curr: trans_neg_indices, self.trans_corr: trans_pos_indices}

      if not trans_init_pos_indices:
        sess.run(self.train, feed_dict)
      else:
        feed_dict[self.trans_init_corr] = trans_init_pos_indices
        feed_dict[self.trans_init_curr] = trans_init_neg_indices
        sess.run(self.train_with_init, feed_dict)

  def fit_ll(self, epochs: int = 100, interval: int = 20):
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(max_to_keep=epochs)
      for epoch in range(1, epochs + 1):
        print('epoch:', epoch)
        for _ in range(self.batch_count):
          characters, labels, lengths = self.get_batch()
          # scores = sess.run(self.output, feed_dict={self.input: characters})
          feed_dict = {self.input: characters, self.real_indices: labels, self.seq_length: lengths}
          sess.run(self.train_ll, feed_dict=feed_dict)
          # self.fit_batch(characters, labels, lengths, sess)
        # if epoch % interval == 0:
        model_path = '../dnlp/models/cws{0}.ckpt'.format(epoch)
        saver.save(sess, model_path)
        self.save_config(model_path)

  def fit_batch_ll(self):
    pass

  def generate_transition_update_index(self, correct_labels, current_labels):
    if correct_labels.shape != current_labels.shape:
      print('sequence length is not equal')
      return None

    before_corr = correct_labels[0]
    before_curr = current_labels[0]
    update_init = False

    trans_init_pos = None
    trans_init_neg = None
    trans_pos = []
    trans_neg = []

    if before_corr != before_curr:
      trans_init_pos = [before_corr]
      trans_init_neg = [before_curr]
      update_init = True

    for _, (corr_label, curr_label) in enumerate(zip(correct_labels[1:], current_labels[1:])):
      if corr_label != curr_label or before_corr != before_curr:
        trans_pos.append([before_corr, corr_label])
        trans_neg.append([before_curr, curr_label])
      before_corr = corr_label
      before_curr = curr_label

    return trans_pos, trans_neg, trans_init_pos, trans_init_neg, update_init

  def predict(self, sentence: str, return_labels=False):
    if self.mode != 'predict':
      raise Exception('mode is not allowed to predict')

    input = self.indices2input(self.sentence2indices(sentence))
    runner = [self.output, self.transition, self.transition_init]
    output, trans, trans_init = self.sess.run(runner, feed_dict={self.input: input})
    labels = self.viterbi(output, trans, trans_init)
    if not return_labels:
      return self.tags2words(sentence, labels)
    else:
      return self.tags2words(sentence, labels), self.tag2sequences(labels)

  def get_embedding_layer(self) -> tf.Tensor:
    embeddings = self.__get_variable([self.dict_size, self.embed_size], 'embeddings')
    self.params.append(embeddings)
    if self.mode == 'train':
      input_size = [self.batch_size, self.batch_length, self.concat_embed_size]
      layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input), input_size)
    else:
      layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input), [1, -1, self.concat_embed_size])
    return layer

  def get_mlp_layer(self, layer: tf.Tensor) -> tf.Tensor:
    hidden_weight = self.__get_variable([self.hidden_units, self.concat_embed_size], 'hidden_weight')
    hidden_bias = self.__get_variable([self.hidden_units, 1, 1], 'hidden_bias')
    self.params += [hidden_weight, hidden_bias]
    layer = tf.sigmoid(tf.tensordot(hidden_weight, layer, [[1], [0]]) + hidden_bias)
    return layer

  def get_rnn_layer(self, layer: tf.Tensor) -> tf.Tensor:
    rnn = tf.nn.rnn_cell.RNNCell(self.hidden_units)
    rnn_output, rnn_out_state = tf.nn.dynamic_rnn(rnn, layer, dtype=self.dtype)
    self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    return tf.transpose(rnn_output)

  def get_lstm_layer(self, layer: tf.Tensor) -> tf.Tensor:
    lstm = tf.nn.rnn_cell.LSTMCell(self.hidden_units)
    lstm_output, lstm_out_state = tf.nn.dynamic_rnn(lstm, layer, dtype=self.dtype)
    self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    return lstm_output

  def get_gru_layer(self, layer: tf.Tensor) -> tf.Tensor:
    gru = tf.nn.rnn_cell.GRUCell(self.hidden_units)
    gru_output, gru_out_state = tf.nn.dynamic_rnn(gru, layer, dtype=self.dtype)
    self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    return tf.transpose(gru_output)

  def get_dropout_layer(self, layer: tf.Tensor) -> tf.Tensor:
    return tf.layers.dropout(layer, self.dropout_rate)

  def get_output_layer(self, layer: tf.Tensor) -> tf.Tensor:
    output_weight = self.__get_variable([self.hidden_units, self.tags_count], 'output_weight')
    output_bias = self.__get_variable([1, 1, self.tags_count], 'output_bias')
    self.params += [output_weight, output_bias]
    return tf.tensordot(layer, output_weight, [[2], [0]]) + output_bias

  def get_loss(self) -> (tf.Tensor, tf.Tensor):
    output_loss = tf.reduce_sum(tf.gather_nd(self.output, self.ll_curr) - tf.gather_nd(self.output, self.ll_corr))
    trans_loss = tf.gather_nd(self.transition, self.trans_curr) - tf.gather_nd(self.transition, self.trans_corr)
    trans_i_curr = tf.gather_nd(self.transition_init, self.trans_init_curr)
    trans_i_corr = tf.gather_nd(self.transition_init, self.trans_init_corr)
    trans_init_loss = tf.reduce_sum(trans_i_curr - trans_i_corr)
    loss = output_loss + trans_loss
    regu = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.lam), self.params)
    l1 = loss + regu
    l2 = l1 + trans_init_loss
    return l1, l2

  def __get_variable(self, size, name) -> tf.Variable:
    return tf.Variable(tf.truncated_normal(size, stddev=1.0 / math.sqrt(size[-1]), dtype=self.dtype), name=name)
