# -*- coding: UTF-8 -*-
import sys
import getopt
from dnlp.config.config import DnnCrfConfig
from dnlp.core.dnn_crf import DnnCrf
from dnlp.utils.evaluation import get_cws_statistics, evaluate_cws


def train_cws():
  data_path = '../dnlp/data/cws/pku_training.pickle'
  config = DnnCrfConfig()
  dnncrf = DnnCrf(config=config, data_path=data_path, nn='lstm')
  dnncrf.fit_ll()


def test_cws():
  sentence = '小明来自南京师范大学'
  model_path = '../dnlp/models/cws1.ckpt'
  config = DnnCrfConfig()
  dnncrf = DnnCrf(config=config, mode='predict', model_path=model_path, nn='lstm')
  res, labels = dnncrf.predict(sentence, return_labels=True)
  print(res)
  evaluate_cws(dnncrf, '../dnlp/data/cws/pku_test.pickle')


if __name__ == '__main__':
  try:
    opts, args = getopt.getopt(sys.argv[1:], 'tp', [])
    if len(opts) != 1:
      raise Exception('cmd args count is not 1')
    opts = opts[0]
    if opts[0] == '-t':
      train_cws()
    elif opts[0] == '-p':
      test_cws()
    else:
      raise Exception('unknown cmd arg')
  except Exception as e:
    print(e.args[0])
