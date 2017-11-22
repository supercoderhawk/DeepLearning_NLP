# -*- coding: UTF-8 -*-
import numpy as np
import pickle
from dnlp.config.config import DnnCrfConfig
from dnlp.utils.constant import BATCH_PAD, STRT_VAL, END_VAL, TAG_PAD, TAG_BEGIN, TAG_INSIDE, TAG_SINGLE


class DnnCrfBase(object):
  def __init__(self, config: DnnCrfConfig=None, data_path: str = '', mode: str = 'train', model_path: str = ''):
    # 加载数据
    self.data_path = data_path
    self.config_suffix = '.config.pickle'
    if mode == 'train':
      self.dictionary, self.tags, self.characters, self.labels = self.__load_data()
    else:
      self.model_path = model_path
      self.config_path = self.model_path + self.config_suffix
      self.dictionary, self.tags = self.__load_config()
    self.tags_count = len(self.tags) - 1  # 忽略TAG_PAD
    self.tags_map = self.__generate_tag_map()
    self.dict_size = len(self.dictionary)
    # 初始化超参数
    self.skip_left = config.skip_left
    self.skip_right = config.skip_right
    self.embed_size = config.embed_size
    self.hidden_units = config.hidden_units
    self.learning_rate = config.learning_rate
    self.lam = config.lam
    self.dropout_rate = config.dropout_rate
    self.windows_size = self.skip_left + self.skip_right + 1
    self.concat_embed_size = self.embed_size * self.windows_size
    self.batch_length = config.batch_length
    self.batch_size = config.batch_size
    # 数据
    if mode == 'train':
      self.sentences_length = list(map(lambda s: len(s), self.characters))
      self.sentences_count = len(self.sentences_length)
      self.batch_count = self.sentences_count // self.batch_size
      self.batch_start = 0

  def __load_data(self) -> (dict, tuple, np.ndarray, np.ndarray):
    with open(self.data_path, 'rb') as f:
      data = pickle.load(f)
      return data['dictionary'], data['tags'], data['characters'], data['labels']

  def __load_config(self) -> (dict, tuple):
    with open(self.config_path, 'rb') as cf:
      config = pickle.load(cf)
      return config['dictionary'], config['tags']

  def save_config(self, model_path: str):
    config_path = model_path + self.config_suffix
    config = {}
    config['dictionary'] = self.dictionary
    config['tags'] = self.tags
    with open(config_path, 'wb') as cf:
      pickle.dump(config, cf)

  def __generate_tag_map(self):
    tags_map = {}
    for i in range(len(self.tags)):
      tags_map[self.tags[i]] = i
    return tags_map

  def get_batch(self) -> (np.ndarray, np.ndarray, np.ndarray):
    if self.batch_start + self.batch_size > self.sentences_count:
      new_start = self.batch_start + self.batch_size - self.sentences_count
      chs_batch = self.characters[self.batch_start:] + self.characters[:new_start]
      lls_batch = self.labels[self.batch_start:] + self.labels[:new_start]
      len_batch = self.sentences_length[self.batch_start:] + self.sentences_length[:new_start]
    else:
      new_start = self.batch_start + self.batch_size
      chs_batch = self.characters[self.batch_start:new_start]
      lls_batch = self.labels[self.batch_start:new_start]
      len_batch = self.sentences_length[self.batch_start:new_start]
    for i, (chs, lls) in enumerate(zip(chs_batch, lls_batch)):
      if len(chs) > self.batch_length:
        chs_batch[i] = chs[:self.batch_length]
        lls_batch[i] = list(map(lambda t: self.tags_map[t], lls[:self.batch_length]))
        len_batch[i] = self.batch_length
      else:
        ext_size = self.batch_length - len(chs)
        chs_batch[i] = chs + ext_size * [self.dictionary[BATCH_PAD]]
        lls_batch[i] = list(map(lambda t: self.tags_map[t], lls)) + ext_size * [0]#[self.tags_map[TAG_PAD]]

    self.batch_start = new_start
    return self.indices2input(chs_batch), np.array(lls_batch, dtype=np.int32), np.array(len_batch, dtype=np.int32)

  def viterbi(self, emission: np.ndarray, transition: np.ndarray, transition_init: np.ndarray):
    length = emission.shape[1]
    path = np.ones([self.tags_count, length], dtype=np.int32) * -1
    corr_path = np.zeros([length], dtype=np.int32)
    path_score = np.ones([self.tags_count, length], dtype=np.float64) * (np.finfo('f').min)
    path_score[:, 0] = transition_init + emission[:, 0]

    for pos in range(1, length):
      for t in range(self.tags_count):
        for prev in range(self.tags_count):
          temp = path_score[prev][pos - 1] + transition[prev][t] + emission[t][pos]
          if temp >= path_score[t][pos]:
            path[t][pos] = prev
            path_score[t][pos] = temp

    max_index = np.argmax(path_score[:, -1])
    corr_path[length - 1] = max_index
    for i in range(length - 1, 0, -1):
      max_index = path[max_index][i]
      corr_path[i - 1] = max_index

    return corr_path

  def sentence2indices(self, sentence: str) -> list:
    return list(map(lambda ch: self.dictionary[ch], sentence))

  def indices2input(self, indices: list) -> np.ndarray:
    res = []
    if isinstance(indices[0], list):
      for idx in indices:
        res.append(self.__indices2input_single(idx))
    else:
      res = self.__indices2input_single(indices)

    return np.array(res, np.int32)

  def __indices2input_single(self, indices: list) -> list:
    ext_indices = [STRT_VAL] * self.skip_left
    ext_indices.extend(indices + [END_VAL] * self.skip_right)
    seq = []
    for index in range(self.skip_left, len(ext_indices) - self.skip_right):
      seq.append(ext_indices[index - self.skip_left: index + self.skip_right + 1])

    return seq

  def tags2words(self, sentence: str, tags_seq: np.ndarray) -> list:
    words = []
    word = ''
    for tag_index, tag in enumerate(tags_seq):
      if tag == self.tags_map[TAG_SINGLE]:
        words.append(sentence[tag_index])
      elif tag == self.tags_map[TAG_BEGIN]:
        word = sentence[tag_index]
      elif tag == self.tags_map[TAG_INSIDE]:
        word += sentence[tag_index]
      else:
        words.append(word + sentence[tag_index])
        word = ''
    # 处理最后一个标记为I的情况
    if word != '':
      words.append(word)

    return words

  def tags2entities(self, sentence: str, tags_seq: np.ndarray, return_start: bool = False):
    entities = []
    entity_starts = []
    entity = ''

    for tag_index, tag in enumerate(tags_seq):
      if tag == 0:
        continue
      elif tag == 1:
        if entity:
          entities.append(entity)
        entity = sentence[tag_index]
        entity_starts.append(tag_index)
      else:
        entity += sentence[tag_index]
    if entity != '':
      entities.append(entity)
    if return_start:
      return entities, entity_starts
    else:
      return entities
