# -*- coding: UTF-8 -*-
import os
from shutil import copyfile
from dnlp.data_process.process_cws import ProcessCWS


def copy():
  src_folder = '../datasets/'
  dst_base_folder = 'dnlp/data/'
  if not os.path.exists(dst_base_folder):
    os.makedirs(dst_base_folder)
  pku = 'pku_training.utf8'
  copyfile(src_folder + pku, dst_base_folder + pku)


def build_cws_datasets():
  files = ('pku_training.utf8',)
  base_folder = 'dnlp/data/cws/'
  if not os.path.exists(base_folder):
    os.makedirs(base_folder)
  ProcessCWS(files=files, base_folder=base_folder, name='pku_training')


if __name__ == '__main__':
  copy()
  build_cws_datasets()
