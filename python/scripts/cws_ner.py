# -*- coding: UTF-8 -*-
import sys
import getopt
from dnlp.config.config import DnnCrfConfig
from dnlp.core.dnn_crf import DnnCrf
from dnlp.utils.evaluation import get_cws_statistics, evaluate_cws


def train_cws():
  data_path = '../dnlp/data/cws/msr_training.pickle'
  config = DnnCrfConfig()
  dnncrf = DnnCrf(config=config, data_path=data_path, nn='bilstm')
  dnncrf.fit()


def test_cws():
  sentence = '小明来自南京师范大学'
  sentence = '中国人民决心继承邓小平同志的遗志，继续把建设有中国特色社会主义事业推向前进。'
  model_path = '../dnlp/models/cws4.ckpt'
  config = DnnCrfConfig()
  dnncrf = DnnCrf(config=config, mode='predict', model_path=model_path, nn='bilstm')
  res, labels = dnncrf.predict_ll(sentence, return_labels=True)
  print(res)
  evaluate_cws(dnncrf, '../dnlp/data/cws/msr_test.pickle')


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
