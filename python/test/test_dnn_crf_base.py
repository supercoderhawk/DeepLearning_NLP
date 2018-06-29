# -*- coding:utf-8 -*-
from unittest import TestCase
from dnlp.core.dnn_crf_base import *
class TestDnnCRFBase(TestCase):
  def test_viterbi(self):
    transition = []
    transtion_init = []
    