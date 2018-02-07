# -*- coding:utf-8 -*-
from dnlp.data_process.processor import Preprocessor
class ProcessConll(Preprocessor):
  def __init__(self,*,files:tuple,name:str,base_folder:str='dnlp/data/',dict_path:str=''):
    if dict_path:
      Preprocessor.__init__(self,base_folder=base_folder, dict_path=dict_path)
    else:
      Preprocessor.__init__(self, base_folder=base_folder, files=files,dict_path=base_folder+name+'.utf8')
