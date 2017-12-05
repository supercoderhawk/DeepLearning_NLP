# -*- coding: UTF-8 -*-
import pickle
from dnlp.data_process.processor import Preprocessor
from dnlp.utils.constant import NER_TAGS


class ProcessNER(Preprocessor):
  def __init__(self, *, files: tuple = (), base_folder: str = 'dnlp/data', dict_path: str = '', name: str = '', mode:str='train'):
    if base_folder == '':
      raise Exception('base folder is empty')
    if dict_path != '':
      Preprocessor.__init__(self, base_folder=base_folder, dict_path=dict_path)
    else:
      if name == '':
        raise Exception('ff')
      Preprocessor.__init__(self, base_folder=base_folder, files=files, dict_path=base_folder + name + '_dict.utf8')
    self.tags = NER_TAGS
    self.files = files
    self.name = name
    self.base_folder = base_folder
    self.sentences, self.labels = self.preprocess()
    if mode == 'train':
      self.sentences = self.map_to_indices()
    self.save_data()

  def preprocess(self):
    sentences = []
    labels = []
    for file in self.files:
      with open(self.base_folder + file, encoding='utf-8') as f:
        entries = [l.split(' ') for l in f.read().splitlines()]
        sentence = ''
        label = []
        for entry in entries:
          if len(entry) == 1:
            sentences.append(sentence)
            labels.append(label)
            sentence = ''
            label = []
          elif len(entry) == 2:
            sentence += entry[0]
            label.append(entry[1])

    return sentences, labels

  def map_to_indices(self):
    sentence_indices = []
    for sentence in self.sentences:
      sentence_indices.append([self.dictionary[c] for c in sentence])
    return sentence_indices

  def save_data(self):
    data = {}
    data['characters'] = self.sentences
    data['labels'] = self.labels
    data['dictionary'] = self.dictionary
    data['tags'] = self.tags
    with open(self.base_folder + self.name + '.pickle', 'wb') as f:
      pickle.dump(data, f)
