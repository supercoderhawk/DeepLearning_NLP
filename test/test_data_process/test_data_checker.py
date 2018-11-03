# -*- coding: utf-8 -*-
from dnlp.data_process.data_checker import *
from dnlp.utils.utils import read_json_file
from dnlp.utils.constant import *
from unittest import TestCase


class TestDataChecker(TestCase):
    def test_check_sequence_json_entry(self):
        data = read_json_file(CWS_PKU_TRAINING_JSON_FILE)
        for idx, entry in enumerate(data):
            ret = check_sequence_json_entry(entry)
            if not ret:
                print(idx, entry)
