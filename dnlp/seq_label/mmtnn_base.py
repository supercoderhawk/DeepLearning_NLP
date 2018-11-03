# -*- coding: UTF-8 -*-
from dnlp.config.seq_label_config import MMTNNConfig


class MMTNNBase(object):
    def __init__(self, *, config: MMTNNConfig, data_path: str = '', mode: str = ''):
        pass

    def __load_data(self):
        pass

    def generate_batch(self):
        pass
