# -*- coding: UTF-8 -*-
from dnlp.utils.constant import BATCH_PAD, BATCH_PAD_VAL, UNK, UNK_VAL, STRT, STRT_VAL, END, END_VAL


class Preprocessor(object):
  def __init__(self, *, base_folder: str, files: tuple = (), dict_path: str = ''):
    self.base_folder = base_folder
    if files != ():
      self.dictionary = self.build_dictionary(files=files, output_dict_path=dict_path)
    else:
      self.dictionary = self.read_dictionary(dict_path)

  def read_dictionary(self, dict_path: str, reverse=False):
    dictionary = {}
    with open(dict_path, encoding='utf8') as d:
      items = d.readlines()
      for item in items:
        pair = item.split(' ')
        dictionary[pair[0]] = int(pair[1])
    if reverse:
      return dictionary, dict(zip(dictionary.values(), dictionary.keys()))
    else:
      return dictionary

  def build_dictionary(self, *, files: tuple = (), output_dict_path: str = '', reverse: bool = False):
    if files == ():
      raise Exception('input is none')

    chs = set('')
    file_content = ''
    for file in files:
      with open(self.base_folder + file, encoding='utf8') as f:
        file_content += f.read()
    chs = chs.union(set(file_content))
    chs = chs.difference(['\r', '\n', ' ', ''])
    dictionary = {BATCH_PAD: BATCH_PAD_VAL, UNK: UNK_VAL, STRT: STRT_VAL, END: END_VAL}
    idx = len(dictionary)
    for ch in chs:
      dictionary[ch] = idx
      idx += 1

    if output_dict_path != '':
      with open(output_dict_path, 'w', encoding='utf8') as o_f:
        for ch, idx in zip(dictionary.keys(), dictionary.values()):
          o_f.write(ch + ' ' + str(idx) + '\n')
    if reverse:
      return dictionary, dict(zip(dictionary.values(), dictionary.keys()))
    else:
      return dictionary

  def preprocess(self):
    raise NotImplementedError('not implement method')

  def map_to_indices(self):
    raise NotImplementedError('not implement method')

  def save_data(self):
    raise NotImplementedError('not implement method')
