#-*- coding: UTF-8 -*-
from dnlp.core.re_cnn import RECNN,RECNNConfig


def train_re_cnn():
  config = RECNNConfig()
  dict_path = '../dnlp/data/emr/emr_word_dict.utf8'
  data_path = '../dnlp/data/emr/train_two.pickle'
  recnn = RECNN(config=config,data_path=data_path,dict_path=dict_path)
  recnn.fit()

def test_re_cnn():
  pass
if __name__ == '__main__':
  train_re_cnn()