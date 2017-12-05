# -*- coding: UTF-8 -*-
import sys
import argparse
from dnlp.config.config import DnnCrfConfig
from dnlp.core.dnn_crf import DnnCrf
from dnlp.core.skip_gram import SkipGram
from dnlp.utils.evaluation import evaluate_cws, evaluate_ner


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


def train_emr():
  data_path = '../dnlp/data/emr/emr_training.pickle'
  embedding_path = '../dnlp/data/emr/emr_skip_gram.npy'
  config = DnnCrfConfig()
  mlpcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='mlp', embedding_path=embedding_path)
  mlpcrf.fit()
  rnncrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='rnn', embedding_path=embedding_path)
  rnncrf.fit()
  lstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='lstm')
  lstmcrf.fit()
  bilstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='bilstm')
  bilstmcrf.fit()
  grulstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='gru')
  grulstmcrf.fit()


def test_emr():
  sentence = '多饮多尿多食'
  config = DnnCrfConfig()
  # dnncrf = DnnCrf(config=config, task='ner', mode='predict', model_path=model_path, nn='lstm')
  # res = dnncrf.predict_ll(sentence)
  # print(res)
  embedding_path = '../dnlp/data/emr/emr_skip_gram.npy'
  mlp_model_path = '../dnlp/models/ner-mlp-50.ckpt'
  rnn_model_path = '../dnlp/models/ner-rnn-50.ckpt'
  lstm_model_path = '../dnlp/models/ner-lstm-50.ckpt'
  bilstm_model_path = '../dnlp/models/ner-bilstm-50.ckpt'
  gru_model_path = '../dnlp/models/ner-gru-50.ckpt'
  mlpcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=mlp_model_path, nn='mlp',
                  embedding_path=embedding_path)
  rnncrf = DnnCrf(config=config, task='ner', mode='predict', model_path=rnn_model_path, nn='rnn',
                  embedding_path=embedding_path)
  lstmcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=lstm_model_path, nn='lstm')
  bilstmcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=bilstm_model_path, nn='bilstm')
  grucrf = DnnCrf(config=config, task='ner', mode='predict', model_path=gru_model_path, nn='gru')
  evaluate_ner(mlpcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(rnncrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(lstmcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(bilstmcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(grucrf, '../dnlp/data/emr/emr_test.pickle')


def train_emr_skipgram():
  base_folder = '../dnlp/data/emr/'
  skipgram = SkipGram(base_folder + 'emr_skip_gram.pickle', base_folder + 'emr_skip_gram')
  skipgram.train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--t', dest='train', action='store_true', default=False)
  parser.add_argument('-p', '--p', dest='predict', action='store_true', default=False)
  args = parser.parse_args(sys.argv[1:])
  train = args.train
  predict = args.predict
  if train and predict:
    print('can\'t train and predict at same time')
    exit(1)
  elif not train and not predict:
    print('don\'t enter mode')
    exit(1)

  if train:
    train_emr()
    # train_emr_skipgram()
  else:
    test_emr()
