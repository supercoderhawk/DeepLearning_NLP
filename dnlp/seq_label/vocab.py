# -*- coding: utf-8 -*-
from dnlp.utils.utils import read_lines_in_file
from dnlp.utils.constant import BATCH_PAD, BATCH_PAD_VAL, UNK, UNK_VAL, BOS, BOS_VAL, EOS, EOS_VAL


class Vocab(object):
    def __init__(self, *, dict_path, label_schema):
        self.__dict_path = dict_path
        self.__dictionary = self.__load_dictionary()
        self.__dict_size = len(self.__dictionary)

        self.__label_schema = label_schema
        self.__label_mapping = self.__get_label_mapping()
        self.__reversed_label_mapping = dict(zip(self.__label_mapping.values(), self.__label_mapping.keys()))

    def __load_dictionary(self):
        dictionary = {}
        pre_token = []
        lines = read_lines_in_file(self.__dict_path)
        for line in lines:
            word, index = line.split(' ')
            index = int(index)
            dictionary[word] = index
        dict_indices = list(dictionary.values())

        if BATCH_PAD_VAL in dict_indices and dictionary[BATCH_PAD] != BATCH_PAD_VAL:
            raise Exception('padding index is occupied.')
        else:
            dictionary[BATCH_PAD] = BATCH_PAD_VAL

        if UNK_VAL in dict_indices and dictionary[UNK] != UNK_VAL:
            raise Exception('UNK index is occupied.')
        else:
            dictionary[UNK] = UNK_VAL

        if BOS_VAL in dict_indices and dictionary[BOS] != BOS_VAL:
            raise Exception('BOS index is occupied.')
        else:
            dictionary[BOS] = BOS_VAL

        if EOS_VAL in dict_indices and dictionary[EOS] != EOS_VAL:
            raise Exception('EOS index is occupied.')
        else:
            dictionary[EOS] = EOS_VAL

        return dictionary

    @property
    def dictionary(self):
        return self.__dictionary

    @property
    def dict_size(self):
        return self.__dict_size

    def __get_label_mapping(self):
        label_mapping = {}
        for index, label in enumerate(self.__label_schema):
            label_mapping[label] = index

        return label_mapping

    @property
    def label_mapping(self):
        return self.__label_mapping

    @property
    def reversed_label_mapping(self):
        return self.__reversed_label_mapping
