# -*- coding: UTF-8 -*-


class DnnCrfConfig(object):
  def __init__(self, *, skip_left: int = 1, skip_right: int = 1, embed_size: int = 100, hidden_units: int = 150,
               learning_rate: float = 0.2, lam: float = 1e-4, dropout_rate: float = 0.2, batch_length: int = 100,
               batch_size=20):
    self.__skip_left = skip_left
    self.__skip_right = skip_right
    self.__embed_size = embed_size
    self.__hidden_units = hidden_units
    self.__learning_rate = learning_rate
    self.__lam = lam
    self.__dropout_rate = dropout_rate
    self.__batch_length = batch_length
    self.__batch_size = batch_size

  @property
  def skip_left(self):
    return self.__skip_left

  @property
  def skip_right(self):
    return self.__skip_right

  @property
  def embed_size(self):
    return self.__embed_size

  @property
  def hidden_units(self):
    return self.__hidden_units

  @property
  def learning_rate(self):
    return self.__learning_rate

  @property
  def lam(self):
    return self.__lam

  @property
  def dropout_rate(self):
    return self.__dropout_rate

  @property
  def batch_length(self):
    return self.__batch_length

  @property
  def batch_size(self):
    return self.__batch_size


class MMTNNConfig(object):
  def __init__(self, *, skip_left: int = 2, skip_right: int = 2, character_embed_size: int = 50,
               label_embed_size: int = 50, hidden_unit: int = 150, learning_rate: float = 0.2, lam: float = 10e-4,
               dropout_rate: float = 0.4, batch_length: int = 150, batch_size: int = 20):
    self.__skip_left = skip_left
    self.__skip_right = skip_right
    self.__character_embed_size = character_embed_size
    self.__label_embed_size = label_embed_size
    self.__hidden_unit = hidden_unit
    self.__learning_rate = learning_rate
    self.__lam = lam
    self.__dropout_rate = dropout_rate
    self.__batch_length = batch_length
    self.__batch_size = batch_size

  @property
  def skip_left(self):
    return self.__skip_left

  @property
  def skip_right(self):
    return self.__skip_right

  @property
  def character_embed_size(self):
    return self.__character_embed_size

  @property
  def label_embed_size(self):
    return self.__label_embed_size

  @property
  def hidden_unit(self):
    return self.__hidden_unit

  @property
  def learning_rate(self):
    return self.__learning_rate

  @property
  def lam(self):
    return self.__lam

  @property
  def dropout_rate(self):
    return self.__dropout_rate

  @property
  def batch_length(self):
    return self.__batch_length

  @property
  def batch_size(self):
    return self.__batch_size
