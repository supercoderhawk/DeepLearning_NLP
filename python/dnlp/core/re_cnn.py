#-*- coding: UTF-8 -*-
import tensorflow as tf
from dnlp.core.re_cnn_base import RECNNBase
from dnlp.config import RECNNConfig

class RECNN(RECNNBase):
  def __init__(self,config:RECNNConfig,dtype:type=tf.float32,dict_path:str='',mode:str='train'):
    RECNNBase.__init__(self,config)
    self.dtype = dtype
    self.mode = mode
    self.dictionary = self.read_dictionary(dict_path)


  def __weight_variable(self, shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=self.dtype)
    return tf.Variable(initial,name=name)