# -*- coding: UTF-8 -*-
import argparse
import sys
import time
import csv
import xlsxwriter
from dnlp.core.re_cnn import RECNN, RECNNConfig

WINDOW_LIST = [(2,), (3,), (4,), (2, 3), (3, 4), (2, 3, 4)]
BASE_FOLDER = '../dnlp/data/emr/'
SKIP_GRAM_PATH = BASE_FOLDER + 'emr_word_light_skip_gram.npy'
CBOW_PATH = BASE_FOLDER + 'emr_word_light_cbow.npy'
TRAIN_DATA_PATH_TWO = BASE_FOLDER + 'train_two.pickle'
TRAIN_DATA_PATH_MULTI = BASE_FOLDER + 'train_multi.pickle'


def train_re_cnn():
  data_path_two = BASE_FOLDER + 'train_two.pickle'
  data_path_two_directed = BASE_FOLDER + 'train_two_directed.pickle'
  data_path_multi = BASE_FOLDER + 'train_multi.pickle'
  data_path_multi_directed = BASE_FOLDER + 'train_multi_directed.pickle'
  embedding_path = BASE_FOLDER + 'emr_word_light_skip_gram.npy'
  cbow_path = BASE_FOLDER + 'emr_word_light_cbow.npy'
  for w in WINDOW_LIST:
    start = time.time()
    train_re_cnn_by_window(w, data_path_two_directed, embedding_path=cbow_path, remark='_cbow_directed')
    train_re_cnn_by_window(w, data_path_two_directed, embedding_path=embedding_path, remark='_skip_gram_directed')
    train_re_cnn_by_window(w, data_path_multi_directed, relation_count=28, embedding_path=cbow_path,
                           remark='_cbow_directed')
    train_re_cnn_by_window(w, data_path_multi_directed, relation_count=28, embedding_path=embedding_path,
                           remark='_skip_gram_directed')
    train_re_cnn_by_window(w, data_path_two_directed, remark='_directed')
    train_re_cnn_by_window(w, data_path_multi_directed, 28, remark='_directed')

    print(time.time() - start)


def train_re_cnn_extra():
  data_path_multi = BASE_FOLDER + 'train_multi.pickle'
  data_path_two = BASE_FOLDER + 'train_two.pickle'
  embedding_path = BASE_FOLDER + 'emr_word_light_cbow.npy'

  remark = '_cbow'
  train_re_cnn_by_window((2, 3, 4), data_path_two, embedding_path=embedding_path, remark=remark)
  train_re_cnn_by_window((3,), data_path_multi, relation_count=28, embedding_path=embedding_path, remark=remark)
  train_re_cnn_by_window((4,), data_path_multi, relation_count=28, embedding_path=embedding_path, remark=remark)


def train_re_cnn_by_window(window_size, data_path, relation_count=2, embedding_path='', remark='_with_embedding'):
  dict_path = '../dnlp/data/emr/emr_merged_word_dict.utf8'
  config = RECNNConfig(window_size=window_size)
  if embedding_path:
    recnn = RECNN(config=config, data_path=data_path, dict_path=dict_path, relation_count=relation_count,
                  embedding_path=embedding_path, remark=remark)
  else:
    recnn = RECNN(config=config, data_path=data_path, dict_path=dict_path, relation_count=relation_count, remark=remark)
  recnn.fit(interval=1)


def test_single_model(window_size, e):
  test_re_cnn_by_window(window_size, e, remark='_directed')


def test_re_cnn(mode='two', remark=''):
  epoch = [10, 10, 10, 10, 10, 10]
  epoch_embedding = [10, 5, 5, 10, 10, 10]
  embedding_path = BASE_FOLDER + 'emr_word_light_skip_gram.npy'
  cbow_path = BASE_FOLDER + 'emr_word_light_cbow.npy'
  if mode == 'two':
    relation_count = 2
    filename = 'two_result{0}'.format(remark)
  else:
    relation_count = 28
    filename = 'multi_result{0}'.format(remark)

  with open('../dnlp/data/emr/{0}.csv'.format(filename), 'w', newline='') as f:
    writer = csv.DictWriter(f, ['name', 'p', 'r', 'f1'])
    writer.writeheader()
    for w, e, ee in zip(WINDOW_LIST, epoch, epoch_embedding):
      p, r, f1 = test_re_cnn_by_window(w, e, mode=mode, relation_count=relation_count, remark='_directed')
      # p, r, f1 = test_re_cnn_by_window(w, e, mode='two', relation_count=2)
      writer.writerow({'name': '_'.join(map(str, w)), 'p': fmt(p), 'r': fmt(r), 'f1': fmt(f1)})
      # if w in [(3,), (4,)]:
      #   p, r, f1 = test_re_cnn_by_window(w, ee, mode=mode, relation_count=relation_count, embedding_path=embedding_path,
      #                                    remark='_with_embedding')
      # else:
      #   p, r, f1 = test_re_cnn_by_window(w, ee, mode=mode, relation_count=relation_count, embedding_path=embedding_path,
      #                                    remark='_with_embedding')
      p, r, f1 = test_re_cnn_by_window(w, e, mode=mode, relation_count=relation_count, embedding_path=cbow_path,
                                       remark='_cbow')
      writer.writerow({'name': '_'.join(map(str, w)), 'p': fmt(p), 'r': fmt(r), 'f1': fmt(f1)})
      p, r, f1 = test_re_cnn_by_window(w, e, mode=mode, relation_count=relation_count, embedding_path=embedding_path,
                                       remark='_skip_gram')
      writer.writerow({'name': '_'.join(map(str, w)), 'p': fmt(p), 'r': fmt(r), 'f1': fmt(f1)})


def get_re_cnn_result(mode='two'):
  if mode == 'two':
    relation_count = 2
  else:
    relation_count = 28
  filename = '../dnlp/data/emr/re_cnn_result_{0}.xlsx'.format(mode)
  workbook = xlsxwriter.Workbook(filename)
  for w in WINDOW_LIST:
    core_name = '_'.join(map(str, w))
    sheet = workbook.add_worksheet(core_name)
    for i in range(1, 51):
      p1, r1, f11 = test_re_cnn_by_window(w, i, mode=mode, relation_count=relation_count, remark='_directed')
      p2, r2, f12 = test_re_cnn_by_window(w, i, mode=mode, relation_count=relation_count, remark='_cbow_directed')
      p3, r3, f13 = test_re_cnn_by_window(w, i, mode=mode, relation_count=relation_count, remark='_skip_gram_directed')
      sheet.write_row(i,0,[fmt(p1),fmt(r1),fmt(f11),fmt(p2),fmt(r2),fmt(f12),fmt(p3),fmt(r3),fmt(f13)])

  workbook.close()


def fmt(n):
  return str('{0:.2f}').format(n * 100)


def test_re_cnn_by_window(window, epoch=20, mode='two', relation_count=2, embedding_path='', remark='_with_embedding'):
  config = RECNNConfig(window_size=window)
  dict_path = '../dnlp/data/emr/emr_merged_word_dict.utf8'
  data_path = '../dnlp/data/emr/test_{0}.pickle'.format(mode)
  # embedding_path = BASE_FOLDER + 'emr_word_skip_gram.npy'
  if embedding_path:
    model_path = '../dnlp/models/re_{0}/'.format(mode) + str(epoch) + '-' + '_'.join(map(str, window)) + \
                 '{0}.ckpt'.format(remark)
  else:
    model_path = '../dnlp/models/re_{0}/'.format(mode) + str(epoch) + '-' + '_'.join(
      map(str, window)) + '{0}.ckpt'.format(remark)
  recnn = RECNN(config=config, data_path=data_path, dict_path=dict_path, mode='test', model_path=model_path,
                relation_count=relation_count, embedding_path=embedding_path, remark=remark)
  return recnn.evaluate()


def test_re_cnn_with_embedding():
  config = RECNNConfig(window_size=(2, 3, 4))
  data_path = '../dnlp/data/emr/test_two.pickle'
  base_folder = '../dnlp/data/emr/'
  dict_path = base_folder + 'emr_merged_word_dict.utf8'
  model_path = '../dnlp/models/re_two/10-2_3_4_cbow.ckpt'
  # embedding_path = base_folder + 'emr_word_skip_gram.npy'
  embedding_path = base_folder + 'emr_word_light_cbow.npy'
  recnn = RECNN(config=config, model_path=model_path, mode='test', data_path=data_path, dict_path=dict_path,
                embedding_path=embedding_path)
  recnn.evaluate()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--t', '-t', dest='train', action='store_true', default=False)
  parser.add_argument('--p', '-p', dest='predict', action='store_true', default=False)
  args = parser.parse_args(sys.argv[1:])
  if args.train:
    train_re_cnn()
    # train_re_cnn_by_window((2,3,4), TRAIN_DATA_PATH_TWO, embedding_path=SKIP_GRAM_PATH, remark='_skip_gram')
    # train_re_cnn_by_window((2, 3, 4), TRAIN_DATA_PATH_TWO, embedding_path=CBOW_PATH, remark='_cbow')
    # train_re_cnn_extra()
    # train_re_cnn_with_embedding()
  else:
    # test_re_cnn()
    # test_re_cnn_by_window((2,),epoch=1,embedding_path=SKIP_GRAM_PATH,remark='_skip_gram')
    # test_re_cnn_by_window((2,), epoch=5, embedding_path=CBOW_PATH, remark='_cbow_directed')
    get_re_cnn_result()
    get_re_cnn_result('multi')
    # test_re_cnn_by_window((3,4), 8, mode='two', relation_count=2, remark='_directed')
    # test_re_cnn_by_window((2,), 16, mode='two', relation_count=2, remark='_directed')
    # test_re_cnn_by_window((2,), 16, mode='two', relation_count=2, remark='_cbow_directed')
    # test_re_cnn_by_window((2,), 16, mode='two', relation_count=2, remark='_skip_gram_directed')
    # test_re_cnn_by_window((2,), 8, mode='two', relation_count=2, remark='_directed')
    # test_re_cnn(remark='_directed')
    # test_re_cnn('multi')
    # test_re_cnn_with_embedding()
    # test_single_model((2, 3, 4), 1)
