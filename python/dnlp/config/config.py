# -*- coding: UTF-8 -*-


class DnnCrfConfig(object):
  def __init__(self, *, skip_left: int = 1, skip_right: int = 1, embed_size: int = 100, hidden_units: int = 150,
               learning_rate: float = 0.2, lam: float = 1e-4,dropout_rate:float=0.2, batch_length: int = 100, batch_size=20):
    self.skip_left = skip_left
    self.skip_right = skip_right
    self.embed_size = embed_size
    self.hidden_units = hidden_units
    self.learning_rate = learning_rate
    self.lam = lam
    self.dropout_rate = dropout_rate
    self.batch_length = batch_length
    self.batch_size = batch_size
