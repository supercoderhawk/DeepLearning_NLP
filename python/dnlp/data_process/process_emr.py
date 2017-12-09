#-*- coding: UTF-8 -*-
import os
class ProcessEMR(object):
  def __init__(self,base_folder:str):
    self.base_folder = base_folder

  def get_files(self):
    files = set()
    for l in os.listdir(self.base_folder):
      files.add(os.path.splitext(l)[0])
    return files

  def read_annotations(self):
    pass