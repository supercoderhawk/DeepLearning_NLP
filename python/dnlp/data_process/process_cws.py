# -*- coding: UTF-8 -*-
import re
import pickle
from dnlp.data_process.processor import Preprocessor
from dnlp.utils.constant import TAG_BEGIN, TAG_INSIDE, TAG_END, TAG_SINGLE,CWS_TAGS


class ProcessCWS(Preprocessor):
  def __init__(self, *, files: tuple = (), dict_path: str = '', base_folder: str = 'dnlp/data', name: str = '',
               delimiter: tuple = ('。')):
    self.SPLIT_CHAR = '  '
    if base_folder == '':
      raise Exception('base folder is empty')
    if dict_path != '':
      Preprocessor.__init__(self, base_folder=base_folder, dict_path=dict_path)
    else:
      if name == '':
        raise Exception('')
      Preprocessor.__init__(self, base_folder=base_folder, files=files, dict_path=base_folder + name + '_dict.utf8')
    self.files = files
    self.name = name
    self.delimiter = delimiter
    self.sentences = self.preprocess()
    self.tags = CWS_TAGS
    self.characters, self.labels = self.map_to_indices()
    self.save_data()

  def preprocess(self):
    sentences = []
    for file in self.files:
      with open(self.base_folder + file, encoding='utf8') as f:
        lines = [l.strip() for l in f.read().splitlines()]
        if self.delimiter != ():
          for d in self.delimiter:
            new_lines = []
            for ls in map(lambda i: i.split(d), lines):
              if not ls[-1]:
                new_lines.extend([l+d for l in ls[:-1]])
              else:
                new_lines.extend([l + d for l in ls])
            lines = new_lines
        sentences += lines

    return sentences

  def map_to_indices(self):
    characters = []
    labels = []
    for sentence in self.sentences:
      sentence = re.sub('[ ]+', self.SPLIT_CHAR, sentence).strip()
      words = sentence.split(self.SPLIT_CHAR)
      chs = []
      lls = []
      for word in words:
        if len(word) == 1:
          chs.append(self.dictionary[word])
          lls.append(TAG_SINGLE)
        elif len(word) == 0:
          raise Exception('word length is zero')
        else:
          chs.extend(map(lambda ch: self.dictionary[ch], word))
          lls.append(TAG_BEGIN)
          lls.extend([TAG_INSIDE] * (len(word) - 2))
          lls.append(TAG_END)
      characters.append(chs)
      labels.append(lls)
    return characters, labels

  def save_data(self):
    data = {}
    data['characters'] = self.characters
    data['labels'] = self.labels
    data['dictionary'] = self.dictionary
    data['tags'] = self.tags
    with open(self.base_folder + self.name + '.pickle', 'wb') as f:
      pickle.dump(data, f)
