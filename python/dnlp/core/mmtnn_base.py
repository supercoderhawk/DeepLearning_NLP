# -*- coding: UTF-8 -*-
from dnlp.config.config import MMTNNConfig


class MMTNNBase(object):
  def __init__(self, *, config: MMTNNConfig, data_path: str = '', mode: str = ''):
    self.data_path = data_path
    self.mode = mode
    # 初始化超参数
    self.skip_left = config.skip_left
    self.skip_right = config.skip_right
    self.character_embed_size = config.character_embed_size
    self.label_embed_size = config.label_embed_size
    self.hidden_unit = config.hidden_unit
    self.learning_rate = config.learning_rate
    self.lam = config.lam
    self.dropout_rate = config.dropout_rate
    self.batch_length = config.batch_length
    self.batch_size = config.batch_size
    self.concat_embed_size = (self.skip_right + self.skip_left + 1) * self.character_embed_size + self.label_embed_size

  def __load_data(self):
    pass

  def generate_batch(self):
    pass
