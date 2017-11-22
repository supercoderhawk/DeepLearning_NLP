# -*- coding: UTF-8 -*-
import pickle
from dnlp.data_process.processor import Preprocessor


class ProcessNER(Preprocessor):
  def __init__(self, *,files:tuple=(),base_folder:str='dnlp/data',dict_path:str='',name:str=''):
    if base_folder == '':
      raise Exception('base folder is empty')
    if dict_path != '':
      Preprocessor.__init__(self, base_folder=base_folder, dict_path=dict_path)
    else:
      if name == '':
        raise Exception('ff')
      Preprocessor.__init__(self, base_folder=base_folder, files=files, dict_path=base_folder + name + '_dict.utf8')

  def preprocess(self):
    pass

  def map_to_indices(self):
    pass

  def save_data(self):
    data = {}
    data['characters'] = self.characters
    data['labels'] = self.labels
    data['dictionary'] = self.dictionary
    data['tags'] = self.tags
    with open(self.base_folder + self.name + '.pickle', 'wb') as f:
      pickle.dump(data, f)