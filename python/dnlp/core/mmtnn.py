#-*- coding: UTF-8 -*-
import math
import tensorflow as tf
from dnlp.core.mmtnn_base import MMTNNBase,MMTNNConfig

class MMTNN(MMTNNBase):
  def __init__(self, config:MMTNNConfig,data_path:str='',dtype:type=tf.float32,mode:str=''):
    MMTNNBase.__init__(self, config=config,data_path=data_path,mode=mode)
    self.dtype = dtype
    self.params = []


  def fit(self):
    pass

  def predict(self, sentence):
    pass

  def get_embedding_layer(self):
    with tf.name_scope('embedding_layer') as embedding_layer:
      embeddings = self.__get_variable([self.dict_size, self.character_embed_size], 'embeddings')
      self.params.append(embeddings)
      if self.mode == 'train':
        input_size = [self.batch_size, self.batch_length, self.concat_embed_size]
        layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input), input_size)
      else:
        layer = tf.reshape(tf.nn.embedding_lookup(embeddings, self.input), [1, -1, self.concat_embed_size])
      return layer

  def get_hidden_layer(self):
    pass

  def get_output_layer(self):
    pass

  def __get_variable(self, size, name) -> tf.Variable:
    return tf.Variable(tf.truncated_normal(size, stddev=1.0 / math.sqrt(size[-1]), dtype=self.dtype), name=name)