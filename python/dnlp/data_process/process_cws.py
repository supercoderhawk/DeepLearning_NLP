# -*- coding: UTF-8 -*-
import re
import pickle
import random
from dnlp.data_process.processor import Preprocessor
from dnlp.utils.constant import TAG_BEGIN, TAG_INSIDE, TAG_END, TAG_SINGLE,CWS_TAGS,UNK_VAL


class ProcessCWS(Preprocessor):
  def __init__(self, *, files: tuple = (), dict_path: str = '', base_folder: str = 'dnlp/data', name: str = '',
               mode:str='train',delimiter: tuple = ('。')):
    self.mode = mode
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
    random.shuffle(sentences)
    random.shuffle(sentences)
    random.shuffle(sentences)
    random.shuffle(sentences)
    random.shuffle(sentences)
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
          if self.mode == 'train':
            chs.append(self.dictionary[word] if self.dictionary.get(word) is not None else UNK_VAL)
          else:
            chs.append(word)
          lls.append(TAG_SINGLE)
        elif len(word) == 0:
          raise Exception('word length is zero')
        else:
          if self.mode == 'train':
            chs.extend(map(lambda ch: self.dictionary[ch] if self.dictionary.get(ch) is not None else UNK_VAL, word))
          else:
            chs.append(word)
          lls.append(TAG_BEGIN)
          lls.extend([TAG_INSIDE] * (len(word) - 2))
          lls.append(TAG_END)
      characters.append(chs)
      labels.append(lls)
    if self.mode == 'test':
      characters = list(map(lambda words:''.join(words),characters))
    return characters, labels

  def save_data(self):
    data = {}
    data['characters'] = self.characters
    data['labels'] = self.labels
    data['dictionary'] = self.dictionary
    data['tags'] = self.tags
    with open(self.base_folder + self.name + '.pickle', 'wb') as f:
      pickle.dump(data, f)
