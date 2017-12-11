# -*- coding: UTF-8 -*-
import sys
import argparse
from dnlp.config import DnnCrfConfig
from dnlp.core.dnn_crf import DnnCrf
from dnlp.core.dnn_crf_emr import DnnCrfEmr
from dnlp.core.word2vec import SkipGram
from dnlp.utils.evaluation import evaluate_cws, evaluate_ner


def train_cws():
  data_path = '../dnlp/data/cws/pku_training.pickle'
  config = DnnCrfConfig()
  dnncrf = DnnCrf(config=config, data_path=data_path, nn='lstm')
  dnncrf.fit()


def test_cws():
  sentence = '小明来自南京师范大学'
  sentence = '中国人民决心继承邓小平同志的遗志，继续把建设有中国特色社会主义事业推向前进。'
  model_path = '../dnlp/models/cws32.ckpt'
  config = DnnCrfConfig()
  dnncrf = DnnCrf(config=config, mode='predict', model_path=model_path, nn='lstm')
  res, labels = dnncrf.predict_ll(sentence, return_labels=True)
  print(res)
  evaluate_cws(dnncrf, '../dnlp/data/cws/pku_test.pickle')

def train_emr_cws():
  data_path = '../dnlp/data/emr/emr_cws.pickle'
  config = DnnCrfConfig()
  dnncrf = DnnCrf(config=config, data_path=data_path, nn='lstm',task='cws',remark='emr_cws')
  dnncrf.fit()

def test_emr_cws():
  config = DnnCrfConfig()
  model_path = '../dnlp/models/cws-lstm-emr_cws-20.ckpt'
  dnncrf = DnnCrf(config=config,model_path=model_path,mode='predict', nn='lstm', task='cws', remark='emr_cws')
  sentences = []
  with open('../dnlp/data/emr/emr.txt',encoding='utf-8') as f:
    sentences = [l for l in f.read().splitlines() if l not in ['',None,'\n','\r',':']]
  content = []
  for sentence in sentences:
    if len(sentence)<=2:
      continue
    words = dnncrf.predict_ll(sentence,return_labels=False)
    content.append(' '.join(words))
  with open('../dnlp/data/emr/emr_words.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(content))

def train_emr_ngram(nn):
  data_path = '../dnlp/data/emr/emr_training.pickle'
  config_bi_bigram = DnnCrfConfig(skip_left=1, skip_right=1)
  config_left_bigram = DnnCrfConfig(skip_left=1, skip_right=0)
  config_right_bigram = DnnCrfConfig(skip_left=0, skip_right=1)
  config_unigram = DnnCrfConfig(skip_left=0, skip_right=0)
  mlpcrf_bi_bigram = DnnCrf(config=config_bi_bigram, task='ner', data_path=data_path, nn=nn, remark='bi_bigram')
  mlpcrf_left_bigram = DnnCrf(config=config_left_bigram, task='ner', data_path=data_path, nn=nn,
                              remark='left_bigram')
  mlpcrf_right_bigram = DnnCrf(config=config_right_bigram, task='ner', data_path=data_path, nn=nn,
                               remark='right_bigram')
  mlpcrf_unigram = DnnCrf(config=config_unigram, task='ner', data_path=data_path, nn=nn, remark='unigram')
  mlpcrf_bi_bigram.fit()
  mlpcrf_left_bigram.fit()
  mlpcrf_right_bigram.fit()
  mlpcrf_unigram.fit()


def test_emr_ngram():
  bi_bigram_model_path = '../dnlp/models/ner-lstm-bi_bigram-50.ckpt'
  config_bi_bigram = DnnCrfConfig(skip_left=1, skip_right=1)
  mlpcrf_bi_bigram = DnnCrf(model_path=bi_bigram_model_path, config=config_bi_bigram, mode='predict', task='ner',
                            nn='lstm')
  evaluate_ner(mlpcrf_bi_bigram, '../dnlp/data/emr/emr_test.pickle')
  left_bigram_model_path = '../dnlp/models/ner-lstm-left_bigram-50.ckpt'
  config_left_bigram = DnnCrfConfig(skip_left=1, skip_right=0)
  mlpcrf_left_bigram = DnnCrf(model_path=left_bigram_model_path, config=config_left_bigram, mode='predict', task='ner',
                            nn='lstm')
  evaluate_ner(mlpcrf_left_bigram, '../dnlp/data/emr/emr_test.pickle')
  right_bigram_model_path = '../dnlp/models/ner-lstm-right_bigram-50.ckpt'
  config_right_bigram = DnnCrfConfig(skip_left=0, skip_right=1)
  mlpcrf_right_bigram = DnnCrf(model_path=right_bigram_model_path, config=config_right_bigram, mode='predict',
                               task='ner',nn='lstm')
  evaluate_ner(mlpcrf_right_bigram, '../dnlp/data/emr/emr_test.pickle')
  unigram_model_path = '../dnlp/models/ner-lstm-unigram-50.ckpt'
  config_unigram = DnnCrfConfig(skip_left=0, skip_right=0)
  mlpcrf_unigram = DnnCrf(model_path=unigram_model_path, config=config_unigram, mode='predict', task='ner',
                            nn='lstm')
  evaluate_ner(mlpcrf_unigram, '../dnlp/data/emr/emr_test.pickle')

def train_emr_dropout(nn):
  data_path = '../dnlp/data/emr/emr_training.pickle'
  config_no_dp= DnnCrfConfig(dropout_rate=0)
  mlpcrf_no_dp = DnnCrf(config=config_no_dp, dropout_position='input',task='ner', data_path=data_path, nn=nn,
                            remark='no_dp')
  mlpcrf_no_dp.fit()
  config_20_dp = DnnCrfConfig(dropout_rate=0.2)
  mlpcrf_20_dp_input = DnnCrf(config=config_20_dp, dropout_position='input', task='ner', data_path=data_path, nn=nn,
                        remark='20_dp_input')
  mlpcrf_20_dp_input.fit()
  mlpcrf_20_dp_hidden = DnnCrf(config=config_20_dp, dropout_position='hidden', task='ner', data_path=data_path,
                              nn=nn,
                              remark='20_dp_hidden')
  mlpcrf_20_dp_hidden.fit()
  config_50_dp = DnnCrfConfig(dropout_rate=0.5)
  mlpcrf_50_dp_input = DnnCrf(config=config_50_dp, dropout_position='input', task='ner', data_path=data_path,
                              nn=nn,
                              remark='50_dp_input')
  mlpcrf_50_dp_input.fit()
  mlpcrf_50_dp_hidden = DnnCrf(config=config_50_dp, dropout_position='hidden', task='ner', data_path=data_path,
                               nn=nn,
                               remark='50_dp_hidden')
  mlpcrf_50_dp_hidden.fit()

def train_emr_old_method():
  data_path = '../dnlp/data/emr/emr_training.pickle'
  config = DnnCrfConfig()
  mlpcrf = DnnCrfEmr(config=config, task='ner', data_path=data_path, nn='rnn')
  mlpcrf.fit(interval=1)

def test_emr_old_method():
  model_path = '../dnlp/models/emr_old/rnn-1.ckpt'
  config = DnnCrfConfig()
  mlpcrf = DnnCrfEmr(config=config, task='ner',mode='predict',model_path=model_path, nn='rnn')

  evaluate_ner(mlpcrf, '../dnlp/data/emr/emr_test.pickle')

def train_emr_random_init():
  data_path = '../dnlp/data/emr/emr_training.pickle'
  config = DnnCrfConfig()
  mlpcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='mlp')
  mlpcrf.fit()
  rnncrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='rnn')
  rnncrf.fit()
  lstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='lstm')
  lstmcrf.fit()
  bilstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='bilstm')
  bilstmcrf.fit()
  grulstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='gru')
  grulstmcrf.fit()

def test_emr_random_init():
  config = DnnCrfConfig()
  mlp_model_path = '../dnlp/models/ner-mlp-50.ckpt'
  rnn_model_path = '../dnlp/models/ner-rnn-50.ckpt'
  lstm_model_path = '../dnlp/models/ner-lstm-50.ckpt'
  bilstm_model_path = '../dnlp/models/ner-bilstm-50.ckpt'
  gru_model_path = '../dnlp/models/ner-gru-50.ckpt'
  mlpcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=mlp_model_path, nn='mlp')
  rnncrf = DnnCrf(config=config, task='ner', mode='predict', model_path=rnn_model_path, nn='rnn')
  lstmcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=lstm_model_path, nn='lstm')
  bilstmcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=bilstm_model_path, nn='bilstm')
  grucrf = DnnCrf(config=config, task='ner', mode='predict', model_path=gru_model_path, nn='gru')
  evaluate_ner(mlpcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(rnncrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(lstmcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(bilstmcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(grucrf, '../dnlp/data/emr/emr_test.pickle')

def train_emr_with_embeddings():
  data_path = '../dnlp/data/emr/emr_training.pickle'
  embedding_path = '../dnlp/data/emr/emr_skip_gram.npy'
  config = DnnCrfConfig()
  mlpcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='mlp', embedding_path=embedding_path)
  # mlpcrf.fit()
  rnncrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='rnn', embedding_path=embedding_path)
  # rnncrf.fit()
  lstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='lstm', embedding_path=embedding_path)
  # lstmcrf.fit()
  bilstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='bilstm', embedding_path=embedding_path)
  bilstmcrf.fit()
  grulstmcrf = DnnCrf(config=config, task='ner', data_path=data_path, nn='gru', embedding_path=embedding_path)
  grulstmcrf.fit()


def test_emr_with_embeddings():
  config = DnnCrfConfig()
  embedding_path = '../dnlp/data/emr/emr_skip_gram.npy'
  mlp_model_path = '../dnlp/models/ner-mlp-embedding-50.ckpt'
  rnn_model_path = '../dnlp/models/ner-rnn-embedding-50.ckpt'
  lstm_model_path = '../dnlp/models/ner-lstm-embedding-50.ckpt'
  bilstm_model_path = '../dnlp/models/ner-bilstm-embedding-50.ckpt'
  gru_model_path = '../dnlp/models/ner-gru-embedding-50.ckpt'
  mlpcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=mlp_model_path, nn='mlp',
                  embedding_path=embedding_path)
  rnncrf = DnnCrf(config=config, task='ner', mode='predict', model_path=rnn_model_path, nn='rnn',
                  embedding_path=embedding_path)
  lstmcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=lstm_model_path, nn='lstm',embedding_path=embedding_path)
  bilstmcrf = DnnCrf(config=config, task='ner', mode='predict', model_path=bilstm_model_path, nn='bilstm',
                     embedding_path=embedding_path)
  grucrf = DnnCrf(config=config, task='ner', mode='predict', model_path=gru_model_path, nn='gru',embedding_path=embedding_path)
  evaluate_ner(mlpcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(rnncrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(lstmcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(bilstmcrf, '../dnlp/data/emr/emr_test.pickle')
  evaluate_ner(grucrf, '../dnlp/data/emr/emr_test.pickle')


def train_emr_skipgram():
  base_folder = '../dnlp/data/emr/'
  skipgram = SkipGram(base_folder + 'emr_skip_gram.pickle', base_folder + 'emr_skip_gram')
  skipgram.train()

def train_emr_word_skipgram():
  base_folder = '../dnlp/data/emr/'
  skipgram = SkipGram(base_folder + 'emr_word_skip_gram.pickle', base_folder + 'emr_word_skip_gram',embed_size=300)
  skipgram.train()
  light_skipgram = SkipGram(base_folder + 'emr_word_light_skip_gram.pickle', base_folder + 'emr_word_light_skip_gram',
                           embed_size=300)
  light_skipgram.train()

def train_emr_word_cbow():
  base_folder = '../dnlp/data/emr/'
  cbow = SkipGram(base_folder + 'emr_word_cbow.pickle', base_folder + 'emr_word_cbow',mode='cbow',embed_size=300,
                  window_size=2)
  cbow.train()
  light_cbow = SkipGram(base_folder + 'emr_word_light_cbow.pickle', base_folder + 'emr_word_light_cbow',mode='cbow',
                           embed_size=300,window_size=2)
  light_cbow.train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--t', dest='train', action='store_true', default=False)
  parser.add_argument('-p', '--p', dest='predict', action='store_true', default=False)
  parser.add_argument('-c', '--c', dest='cws', action='store_true', default=False)
  parser.add_argument('-e', '--e', dest='emr', action='store_true', default=False)
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
    if args.cws:
      train_cws()
    elif args.emr:
      # train_emr_old_method()
      # train_emr_cws()
      # train_emr_word_skipgram()
      # train_emr_word_cbow()
      # train_emr_with_embeddings()
      train_emr_ngram('mlp')
      train_emr_ngram('rnn')
      train_emr_ngram('bilstm')
      train_emr_ngram('gru')
      train_emr_dropout('mlp')
      train_emr_dropout('rnn')
      train_emr_dropout('lstm')
      train_emr_dropout('bilstm')
      train_emr_dropout('gru')
      # train_emr_random_init()
      # train_emr_skipgram()
  else:
    if args.cws:
      test_cws()
    elif args.emr:
      test_emr_cws()
      # test_emr_old_method()
      # test_emr_ngram()
      # test_emr_random_init()
      # print('embedding')
      # test_emr_with_embeddings()
