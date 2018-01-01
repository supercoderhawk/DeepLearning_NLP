# -*- coding:utf-8 -*-
import numpy as np
import pickle
from dnlp.config import RECNNConfig
from dnlp.utils.constant import BATCH_PAD, BATCH_PAD_VAL


class RECNNBase(object):
  def __init__(self, config: RECNNConfig, dict_path: str, data_path: str = ''):
    self.window_size = config.window_size
    self.filter_size = config.filter_size
    self.learning_rate = config.learning_rate
    self.dropout_rate = config.dropout_rate
    self.lam = config.lam
    self.word_embed_size = config.word_embed_size
    self.position_embed_size = config.position_embed_size
    self.batch_length = config.batch_length
    self.batch_size = config.batch_size
    self.dictionary = self.read_dictionary(dict_path)
    self.words_size = len(self.dictionary)

  def read_dictionary(self, dict_path):
    with open(dict_path, encoding='utf-8') as f:
      content = f.read().splitlines()
      dictionary = {}
      dict_arr = map(lambda item: item.split(' '), content)
      for _, dict_item in enumerate(dict_arr):
        dictionary[dict_item[0]] = int(dict_item[1])

      return dictionary

  def load_data(self):
    primary = []
    secondary = []
    words = []
    labels = []
    with open(self.data_path, 'rb') as f:
      data = pickle.load(f)
      for sentence in data:
        sentence_words = sentence['words']
        if len(sentence_words) < self.batch_length:
          sentence_words += [self.dictionary[BATCH_PAD]] * (self.batch_length - len(sentence_words))
        else:
          sentence_words = sentence_words[:self.batch_length]
        words.append(sentence_words)
        primary.append(np.arange(self.batch_length) - sentence['primary'] + self.batch_length - 1)
        secondary.append(np.arange(self.batch_length) - sentence['secondary'] + self.batch_length - 1)
        sentence_labels = np.zeros([self.relation_count])
        sentence_labels[sentence['type']] = 1
        labels.append(sentence_labels)
    return np.array(words, np.int32), np.array(primary, np.int32), np.array(secondary, np.int32), np.array(labels,np.float32)
