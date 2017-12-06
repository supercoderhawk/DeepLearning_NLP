# -*- coding: UTF-8 -*-
import os
from shutil import copyfile
from dnlp.data_process.process_cws import ProcessCWS
from dnlp.data_process.process_ner import ProcessNER
from dnlp.data_process.process_character_embedding_pretrain import CharacterEmbeddingPertrainProcess


def init():
  model_path = '../dnlp/models/'
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  log_path = '../dnlp/data/logs/'
  if not os.path.exists(log_path):
    os.makedirs(log_path)


def copy():
  src_folder = '../../datasets/'
  dst_base_folder = '../dnlp/data/cws/'
  if not os.path.exists(dst_base_folder):
    os.makedirs(dst_base_folder)
  files = ['pku_training.utf8', 'pku_test.utf8', 'msr_training.utf8', 'msr_test.utf8']
  for f in files:
    copyfile(src_folder + f, dst_base_folder + f)


def build_cws_datasets():
  base_folder = '../dnlp/data/cws/'
  if not os.path.exists(base_folder):
    os.makedirs(base_folder)
  ProcessCWS(files=('pku_training.utf8',), base_folder=base_folder, name='pku_training')
  ProcessCWS(files=('msr_training.utf8',), base_folder=base_folder, name='msr_training')
  pku_dict_path = base_folder + 'pku_training_dict.utf8'
  ProcessCWS(files=('pku_test.utf8',), dict_path=pku_dict_path, base_folder=base_folder, name='pku_test', mode='test')
  msr_dict_path = base_folder + 'msr_training_dict.utf8'
  ProcessCWS(files=('msr_test.utf8',), dict_path=msr_dict_path, base_folder=base_folder, name='msr_test', mode='test')


def build_emr_datasets():
  base_folder = '../dnlp/data/emr/'
  if not os.path.exists(base_folder):
    os.makedirs(base_folder)
  dict_path = base_folder + 'emr_dict.utf8'
  ProcessNER(files=('emr_training.conll',), dict_path=dict_path, base_folder=base_folder, name='emr_training')
  ProcessNER(files=('emr_test.conll',), dict_path=dict_path, base_folder=base_folder, name='emr_test', mode='test')
  CharacterEmbeddingPertrainProcess(base_folder,('emr.txt',),dict_path,1)

if __name__ == '__main__':
  # init()
  # copy()
  # build_cws_datasets()
  build_emr_datasets()
