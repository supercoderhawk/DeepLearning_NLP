# -*- coding: UTF-8 -*-
import argparse
import sys
import time
import csv
from dnlp.core.re_cnn import RECNN, RECNNConfig

WINDOW_LIST = [(2,),(3,),(4,),(2,3),(3,4),(2,3,4)]
BASE_FOLDER = '../dnlp/data/emr/'
def train_re_cnn():
  data_path_two = BASE_FOLDER + 'train_two.pickle'
  data_path_multi = BASE_FOLDER + 'train_multi.pickle'
  embedding_path = BASE_FOLDER + 'emr_word_skip_gram.npy'
  for w in WINDOW_LIST:
    start = time.time()
    train_re_cnn_by_window(w,data_path_two,embedding_path=embedding_path)
    train_re_cnn_by_window(w, data_path_multi, relation_count=28,embedding_path=embedding_path)
    # train_re_cnn_by_window(w,data_path_two)
    # train_re_cnn_by_window(w, data_path_multi,28)
    print(time.time()-start)
def train_re_cnn_extra():
  data_path_multi = BASE_FOLDER + 'train_multi.pickle'
  embedding_path = BASE_FOLDER + 'emr_word_cbow.npy'
  remark='_cbow'
  train_re_cnn_by_window((3,), data_path_multi, relation_count=28, embedding_path=embedding_path,remark=remark)
  train_re_cnn_by_window((4,), data_path_multi, relation_count=28, embedding_path=embedding_path,remark=remark)

def train_re_cnn_by_window(window_size,data_path,relation_count=2,embedding_path='',remark='_with_embedding'):
  dict_path = '../dnlp/data/emr/emr_merged_word_dict.utf8'
  config = RECNNConfig(window_size=window_size)
  if embedding_path:
    recnn = RECNN(config=config, data_path=data_path, dict_path=dict_path,relation_count=relation_count,
                  embedding_path=embedding_path,remark=remark)
  else:
    recnn = RECNN(config=config, data_path=data_path, dict_path=dict_path, relation_count=relation_count)
  recnn.fit()


def test_re_cnn():
  epoch = [10,10,5,10,10,10]
  epoch_embedding = [10, 5, 5, 10, 10, 10]
  embedding_path = BASE_FOLDER + 'emr_word_cbow.npy'

  with open('../dnlp/data/emr/result.csv','w',newline='') as f:
    writer = csv.DictWriter(f,['name','p','r','f1'])
    writer.writeheader()
    for w,e,ee in zip(WINDOW_LIST,epoch,epoch_embedding):
      p,r,f1 = test_re_cnn_by_window(w,ee,mode='multi',relation_count=28)
      # p, r, f1 = test_re_cnn_by_window(w, e, mode='two', relation_count=2)
      writer.writerow({'name': 'two_' + '_'.join(map(str, w)), 'p': fmt(p), 'r': fmt(r), 'f1': fmt(f1)})
      if w in [(3,),(4,)]:
        p, r, f1 = test_re_cnn_by_window(w, e, mode='multi', relation_count=28, embedding_path=embedding_path,
                                         remark='_cbow')
      else:
        p, r, f1 = test_re_cnn_by_window(w, e,mode='multi',relation_count=28,embedding_path=embedding_path)
      # p, r, f1 = test_re_cnn_by_window(w, e, mode='two', relation_count=2, embedding_path=embedding_path)
      writer.writerow({'name':'two_'+'_'.join(map(str,w)),'p':fmt(p),'r':fmt(r),'f1':fmt(f1)})

def fmt(n):
  return str('{0:.2f}').format(n*100)

def test_re_cnn_by_window(window, epoch=20, mode='two', relation_count=2,embedding_path='',remark='_with_embedding'):
  config = RECNNConfig(window_size=window)
  dict_path = '../dnlp/data/emr/emr_merged_word_dict.utf8'
  data_path = '../dnlp/data/emr/test_{0}.pickle'.format(mode)
  # embedding_path = BASE_FOLDER + 'emr_word_skip_gram.npy'
  if embedding_path:
    model_path = '../dnlp/models/re_{0}/'.format(mode) + str(epoch) + '-' + '_'.join(map(str, window)) + \
                 '{0}.ckpt'.format(remark)
  else:
    model_path = '../dnlp/models/re_{0}/'.format(mode) + str(epoch) + '-' + '_'.join(map(str, window)) + '.ckpt'
  recnn = RECNN(config=config, data_path=data_path, dict_path=dict_path, mode='test', model_path=model_path,
                relation_count=relation_count,embedding_path=embedding_path)
  return recnn.evaluate()


def test_re_cnn_with_embedding():
  config = RECNNConfig()
  data_path = '../dnlp/data/emr/test_two.pickle'
  base_folder = '../dnlp/data/emr/'
  dict_path = base_folder + 'emr_merged_word_dict.utf8'
  model_path = '../dnlp/models/re_two/10-3_4.ckpt'
  embedding_path = base_folder + 'emr_word_skip_gram.npy'
  recnn = RECNN(config=config, model_path=model_path, mode='test', data_path=data_path, dict_path=dict_path,
                embedding_path=embedding_path)
  recnn.evaluate()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--t', '-t', dest='train', action='store_true', default=False)
  parser.add_argument('--p', '-p', dest='predict', action='store_true', default=False)
  args = parser.parse_args(sys.argv[1:])
  if args.train:
    # train_re_cnn()
    train_re_cnn_extra()
    # train_re_cnn_with_embedding()
  else:
    test_re_cnn()
    # test_re_cnn_with_embedding()
