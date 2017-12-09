# -*- coding:utf-8 -*-
from dnlp.config import RECNNConfig
class RECNNBase(object):
  def __init__(self, config:RECNNConfig):
    self.window_size = config.window_size
    self.filter_size = config.filter_size
    self.learning_rate = config.learning_rate
    self.dropout_rate = config.dropout_rate
    self.lam = config.lam
    self.word_embed_size = config.word_embed_size
    self.position_embed_size = config.position_embed_size
    self.batch_length = config.batch_length
    self.batch_size = config.batch_size

  def read_dictionary(self,dict_path):
    with open(dict_path,encoding='utf-8') as f:
      content = f.read().splitlines()
      dictionary = {}
      dict_arr = map(lambda item: item.split(' '), content)
      for _, dict_item in enumerate(dict_arr):
        dictionary[dict_item[0]] = int(dict_item[1])

      return dictionary

