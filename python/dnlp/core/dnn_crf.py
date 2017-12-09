# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import math
import os
from dnlp.core.dnn_crf_base import DnnCrfBase
from dnlp.config import DnnCrfConfig


class DnnCrf(DnnCrfBase):
  def __init__(self, *, config: DnnCrfConfig = None, task='cws', data_path: str = '', dtype: type = tf.float32,
               mode: str = 'train', dropout_position: str = 'input', predict: str = 'll', nn: str, model_path: str = '',
               embedding_path: str = '', remark: str = ''):
    if mode not in ['train', 'predict']:
      raise Exception('mode error')
    if nn not in ['mlp', 'rnn', 'lstm', 'bilstm', 'gru']:
      raise Exception('name of neural network entered is not supported')

    DnnCrfBase.__init__(self, config, data_path, mode, model_path)
    self.dtype = dtype
    self.mode = mode
    self.task = task
    self.nn = nn
    self.remark = remark
    self.embedding_path = embedding_path
    self.graph = tf.Graph()
    with self.graph.as_default():
      # 构建
      # tf.reset_default_graph()
      self.transition = self.__get_variable([self.tags_count, self.tags_count], 'transition')
      self.transition_init = self.__get_variable([self.tags_count], 'transition_init')
      self.params = [self.transition, self.transition_init]
      # 输入层

      if mode == 'train':
        self.input = tf.placeholder(tf.int32, [self.batch_size, self.batch_length, self.windows_size])
        self.real_indices = tf.placeholder(tf.int32, [self.batch_size, self.batch_length])
      else:
        self.input = tf.placeholder(tf.int32, [None, self.windows_size])

      self.seq_length = tf.placeholder(tf.int32, [None])

      # 查找表层
      self.embedding_layer = self.get_embedding_layer()
      # 执行drpout
      if dropout_position == 'input':
        self.embedding_layer = self.get_dropout_layer(self.embedding_layer)
      # 隐藏层
      if nn == 'mlp':
        self.hidden_layer = self.get_mlp_layer(self.embedding_layer)
      elif nn == 'lstm':
        self.hidden_layer = self.get_lstm_layer(self.embedding_layer)
      elif nn == 'bilstm':
        self.hidden_layer = self.get_bilstm_layer(self.embedding_layer)
      elif nn == 'gru':
        self.hidden_layer = self.get_gru_layer(self.embedding_layer)
      else:
        self.hidden_layer = self.get_rnn_layer(self.embedding_layer)
      if dropout_position == 'hidden':
        self.hidden_layer = self.get_dropout_layer(self.hidden_layer)
      # 输出层
      self.output = self.get_output_layer(self.hidden_layer)

      if mode == 'predict':
        if predict != 'll':
          self.output = tf.squeeze(tf.transpose(self.output), axis=2)
        self.seq, self.best_score = tf.contrib.crf.crf_decode(self.output, self.transition, self.seq_length)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(save_path=self.model_path, sess=self.sess)
      else:
        self.crf_loss, _ = tf.contrib.crf.crf_log_likelihood(self.output, self.real_indices, self.seq_length,
                                                         self.transition)
        #self.loss = -self.loss
        self.regularization = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.lam),
                                                                       self.params )
        self.loss = -self.crf_loss/self.batch_size + self.regularization
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        self.new_optimizer = tf.train.AdamOptimizer()
        gvs = self.optimizer.compute_gradients(self.loss)
        cliped_grad = [(tf.clip_by_norm(grad, 5) if grad is not None else grad, var) for grad, var in gvs]
        self.train = self.optimizer.apply_gradients(cliped_grad)  # self.optimizer.minimize(self.loss)
        # self.train = self.optimizer.minimize(self.loss)
        current_dir = os.path.dirname(__file__)
        dest_dir = os.path.realpath(os.path.join(current_dir, '..\\data\\logs'))
        self.train_writer = tf.summary.FileWriter(dest_dir, flush_secs=10)
        self.mean_loss = tf.reduce_mean(self.loss)
        tf.summary.scalar('loss', self.mean_loss)
        self.merged = tf.summary.merge_all()

  def fit(self, epochs: int = 50, interval: int = 10):
    with tf.Session(graph=self.graph) as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(max_to_keep=epochs)
      for epoch in range(1, epochs + 1):
        print('epoch:', epoch)
        j = 0
        for i in range(self.batch_count):
          characters, labels, lengths = self.get_batch()
          feed_dict = {self.input: characters, self.real_indices: labels, self.seq_length: lengths}
          _, summary, loss = sess.run([self.train, self.merged, self.mean_loss], feed_dict=feed_dict)
          self.train_writer.add_summary(summary, j)
          j += 1
        if epoch % interval == 0:
          if not self.embedding_path:
            if self.remark:
              model_path = '../dnlp/models/{0}-{1}-{2}-{3}.ckpt'.format(self.task, self.nn, self.remark, epoch)
            else:
              model_path = '../dnlp/models/{0}-{1}-{2}.ckpt'.format(self.task, self.nn, epoch)
          else:
            model_path = '../dnlp/models/{0}-{1}-embedding-{2}.ckpt'.format(self.task, self.nn, epoch)
          saver.save(sess, model_path)
          self.save_config(model_path)
      self.train_writer.close()

  def predict(self, sentence: str, return_labels=False):
    if self.mode != 'predict':
      raise Exception('mode is not allowed to predict')

    input = self.indices2input(self.sentence2indices(sentence))
    runner = [self.output, self.transition, self.transition_init]
    output, trans, trans_init = self.sess.run(runner, feed_dict={self.input: input})
    labels = self.viterbi(output, trans, trans_init)
    if self.task == 'cws':
      result = self.tags2words(sentence, labels)
    else:
      result = self.tags2entities(sentence, labels)
    if not return_labels:
      return result
    else:
      return result, self.tag2sequences(labels)

  def predict_ll(self, sentence: str, return_labels=False):
    if self.mode != 'predict':
      raise Exception('mode is not allowed to predict')

    input = self.indices2input(self.sentence2indices(sentence))
    runner = [self.seq, self.best_score, self.output, self.transition]
    labels, best_score, output, trans = self.sess.run(runner,
                                                      feed_dict={self.input: input, self.seq_length: [len(sentence)]})
    # print(output)
    # print(trans)
    labels = np.squeeze(labels)
    if self.task == 'cws':
      result = self.tags2words(sentence, labels)
    else:
      result = self.tags2entities(sentence, labels)
    if return_labels:
      return result, self.tag2sequences(labels)
    else:
      return result

  def get_embedding_layer(self) -> tf.Tensor:
    if self.embedding_path:
      embeddings = tf.Variable(np.load(self.embedding_path), trainable=True, name='embeddings')
    else:
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
    layer = tf.sigmoid(tf.tensordot(hidden_weight, tf.transpose(layer), [[1], [0]]) + hidden_bias)
    return tf.transpose(layer)

  def get_rnn_layer(self, layer: tf.Tensor) -> tf.Tensor:
    rnn = tf.nn.rnn_cell.BasicRNNCell(self.hidden_units)
    rnn_output, rnn_out_state = tf.nn.dynamic_rnn(rnn, layer, dtype=self.dtype)
    self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    return rnn_output

  def get_lstm_layer(self, layer: tf.Tensor) -> tf.Tensor:
    lstm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units)
    lstm_output, lstm_out_state = tf.nn.dynamic_rnn(lstm, layer, dtype=self.dtype)
    self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    return lstm_output

  def get_bilstm_layer(self, layer: tf.Tensor) -> tf.Tensor:
    lstm_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units // 2)
    lstm_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_units // 2)
    bilstm_output, bilstm_output_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw, lstm_bw, layer, self.seq_length,
                                                                         dtype=self.dtype)
    self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    return tf.concat([bilstm_output[0], bilstm_output[1]], -1)

  def get_gru_layer(self, layer: tf.Tensor) -> tf.Tensor:
    gru = tf.nn.rnn_cell.GRUCell(self.hidden_units)
    gru_output, gru_out_state = tf.nn.dynamic_rnn(gru, layer, dtype=self.dtype)
    self.params += [v for v in tf.global_variables() if v.name.startswith('rnn')]
    return gru_output

  def get_dropout_layer(self, layer: tf.Tensor) -> tf.Tensor:
    return tf.layers.dropout(layer, self.dropout_rate)

  def get_output_layer(self, layer: tf.Tensor) -> tf.Tensor:
    output_weight = self.__get_variable([self.hidden_units, self.tags_count], 'output_weight')
    output_bias = self.__get_variable([1, 1, self.tags_count], 'output_bias')
    self.params += [output_weight, output_bias]
    return tf.tensordot(layer, output_weight, [[2], [0]]) + output_bias

  def __get_variable(self, size, name) -> tf.Variable:
    return tf.Variable(tf.truncated_normal(size, stddev=1.0 / math.sqrt(size[-1]), dtype=self.dtype), name=name)
