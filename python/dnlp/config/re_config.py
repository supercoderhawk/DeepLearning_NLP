# -*- coding:utf-8 -*-

class RECNNConfig(object):
  def __init__(self,window_size:tuple=(3,4,),filter_size:int=150,learning_rate:float=0.05,dropout_rate:float=0.5,
               lam:float=5e-4,word_embed_size:int=300,position_embed_size:int=50,batch_length:int=85,
               batch_size:int=20):
    self.__window_size = window_size
    self.__filter_size = filter_size
    self.__learning_rate = learning_rate
    self.__dropout_rate = dropout_rate
    self.__lam = lam
    self.__word_embed_size = word_embed_size
    self.__position_embed_size = position_embed_size
    self.__batch_length = batch_length
    self.__batch_size = batch_size

  @property
  def window_size(self):
    return self.__window_size

  @property
  def filter_size(self):
    return self.__filter_size

  @property
  def learning_rate(self):
    return self.__learning_rate

  @property
  def dropout_rate(self):
    return self.__dropout_rate

  @property
  def lam(self):
    return self.__lam

  @property
  def word_embed_size(self):
    return self.__word_embed_size

  @property
  def position_embed_size(self):
    return self.__position_embed_size

  @property
  def batch_length(self):
    return self.__batch_length

  @property
  def batch_size(self):
    return self.__batch_size
