# -*- coding:utf-8 -*-
from dnlp.config.sequence_labeling_config import DnnCrfConfig
from dnlp.core.dnn_crf import DnnCrf
def ner(sentence):
  data_path = ''
  config_bi_bigram = DnnCrfConfig(skip_left=0, skip_right=0)
  lstmcrf = DnnCrf(config=config_bi_bigram, task='ner', data_path=data_path, nn='lstm', remark='lstm')
  return lstmcrf.predict(sentence)
def cws(sentence):
  config = DnnCrfConfig()
  model_path = '../dnlp/models/emr/cws-lstm-emr_cws-20.ckpt'
  dnncrf = DnnCrf(config=config, model_path=model_path, mode='predict', nn='lstm', task='cws', remark='emr_cws')
  return dnncrf.predict(sentence)

def prepare_rel(sentence):
  cws_res = cws(sentence)


def rel():
  pass

def get_sentences(filename):
  with open('../dnlp/data/emr/emr_paper/train/'+filename,encoding='utf-8') as f:
    return f.read().split('。')

if __name__ == '__main__':
  sentences = get_sentences('996716_admission.txt')
  for sentence in sentences:
    prepare_rel(sentence)