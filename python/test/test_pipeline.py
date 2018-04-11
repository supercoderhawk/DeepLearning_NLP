from unittest import TestCase

from scripts.pipeline import modify_cws
# -*- coding:utf-8 -*-
class TestPipeline(TestCase):
  def test_modify_cws(self):
    # self.fail()
    cws_res = ['小明来自','上海','理工大学']
    res = modify_cws(cws_res,'工大学',7)
    print(res)
