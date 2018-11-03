# -*- coding: utf-8 -*-
"""Sequence labeling data provider in batch"""
import numpy as np
from dnlp.utils.utils import read_conll_file
from dnlp.utils.constant import BATCH_PAD, BOS, EOS


class SequenceData(object):
    """Provides the sequence labeling data that can be feed into neural network from conll file.

    """

    def __init__(self, *, filename, word_mapper, label_mapper, sent_padding_length,
                 is_skip_window=False, skip_left=0, skip_right=0):
        """Creates the sequence labeling data object

        :param filename: conll filename
        :param word_mapper:
        :param label_mapper:
        :param sent_padding_length:
        :param is_skip_window:
        :param skip_left:
        :param skip_right:
        """
        self.__filename = filename
        self.__word2id_mapper = word_mapper
        self.__label2id_mapper = label_mapper
        self.__sent_padding_length = sent_padding_length
        self.__is_skip_window = is_skip_window
        self.__skip_left = skip_left
        self.__skip_right = skip_right
        self.__offset = 0
        self.__sent_count = 0
        self.__sents, self.__labels, self.__seq_lengths = self.__transform()

    def __transform(self):
        sents = []
        labels = []
        seq_lengths = []
        for sent_words, sent_labels in read_conll_file(self.__filename):
            mapped_words = [self.__word2id_mapper[word] for word in sent_words]
            mapped_labels = [self.__label2id_mapper[label] for label in sent_labels]
            if len(mapped_words) >= self.__sent_padding_length:
                mapped_words = mapped_words[:self.__sent_padding_length]
                mapped_labels = mapped_labels[:self.__sent_padding_length]
            else:
                pad_idx = self.__word2id_mapper[BATCH_PAD]
                mapped_words += [pad_idx] * (self.__sent_padding_length - len(sent_words))
                mapped_labels += [0] * (self.__sent_padding_length - len(sent_labels))
            if self.__is_skip_window:
                sents.append(self.__indices2index_windows(mapped_words))
            else:
                sents.append(mapped_words)
            labels.append(mapped_labels)
            seq_lengths.append(len(sent_labels))
            self.__sent_count += 1
        return np.array(sents), np.array(labels), np.array(seq_lengths)

    def __indices2index_windows(self, seq_indices):
        ext_indices = [self.__word2id_mapper[BOS]] * self.__skip_left
        ext_indices.extend(seq_indices + [self.__word2id_mapper[EOS]] * self.__skip_right)
        seq = []
        for index in range(self.__skip_left, len(ext_indices) - self.__skip_right):
            seq.append(ext_indices[index - self.__skip_left: index + self.__skip_right + 1])

        return seq

    def __get_batch(self, batch_size):
        if self.__offset + batch_size <= self.__sent_count:
            s = slice(self.__offset, self.__offset + batch_size)
            self.__offset += batch_size
            return self.__sents[s], self.__labels[s], self.__seq_lengths[s]
        else:
            s1 = slice(self.__offset, self.__sent_count)
            s2 = slice(0, self.__offset + batch_size - self.__sent_count)
            sents = np.concatenate((self.__sents[s1], self.__sents[s2]))
            labels = np.concatenate((self.__labels[s1], self.__labels[s2]))
            seq_lengths = np.concatenate((self.__seq_lengths[s1], self.__seq_lengths[s2]))
            self.__offset += batch_size - self.__sent_count
            return sents, labels, seq_lengths

    def mini_batch(self, batch_size):
        while True:
            yield self.__get_batch(batch_size)

    @property
    def sent_count(self):
        return self.__sent_count
