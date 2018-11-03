# -*- coding: utf-8 -*-
from unittest import TestCase
from dnlp.utils.utils import change_extname


class TestUtils(TestCase):
    def test_change_extname(self):
        src_1 = 'filename.txt'
        dest_1 = 'filename.json'
        self.assertEqual(change_extname(src_1, '.json'), dest_1)
        src_2 = 'filename'
        dest_2 = 'filename.rar'
        self.assertEqual(change_extname(src_2, '.rar'), dest_2)
