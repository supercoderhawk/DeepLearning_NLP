# -*- coding: utf-8 -*-
"""Base class of NNCRF, provide data loader,checker and viterbi"""
import tensorflow as tf
import numpy as np
from dnlp.config.seq_label_config import NeuralNetworkCRFConfig
from .seq_data import SequenceData
from .vocab import Vocab
from dnlp.utils.constant import BATCH_PAD, BOS, EOS, UNK, MODE_FIT, MODE_INFERENCE


class NeuralNetworkCRFBase(object):
    model_name = 'NeuralNetworkCRF'
    model_description = 'Neural Network CRF sequence labeling model'

    def __init__(self, *, mode: str, config: NeuralNetworkCRFConfig):
        if mode not in (MODE_FIT, MODE_INFERENCE):
            raise Exception('{0} mode error'.format(NeuralNetworkCRFBase.model_name))
        self.config = config
        self.mode = mode
        self.__configuration_checker()

        self.vocab = Vocab(dict_path=self.config.dict_path,
                           label_schema=self.config.label_schema)
        self.dictionary = self.vocab.dictionary
        self.label_schema = self.config.label_schema
        self.label_mapping = self.vocab.label_mapping
        self.label_count = len(self.label_mapping)
        self.reversed_label_mapping = self.vocab.reversed_label_mapping
        self.dict_size = self.vocab.dict_size

        tf.reset_default_graph()
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        if self.mode == MODE_FIT:
            self.data = SequenceData(filename=self.config.training_filename,
                                     word_mapper=self.dictionary,
                                     label_mapper=self.label_mapping,
                                     sent_padding_length=self.config.batch_length,
                                     is_skip_window=True,
                                     skip_left=self.config.skip_left,
                                     skip_right=self.config.skip_right)
            self.batch_count = self.data.sent_count // self.config.batch_size

    def __configuration_checker(self):
        if not len(self.config.hidden_layers):
            raise Exception('hidden layer can\'t be empty!')
        elif type(self.config.hidden_layers) not in {list, tuple}:
            raise Exception('hidden layer config must be list or tuple')
        for layer in self.config.hidden_layers:
            if not isinstance(layer, dict):
                raise Exception('')

    def sentences2input_indices(self, sentences, padding_length):
        input_indices = []
        for sentence in sentences:
            indices = []
            for word in sentence:
                if word in self.dictionary:
                    index = self.dictionary[word]
                else:
                    index = self.dictionary[UNK]
                indices.append(index)
            input_indices.append(self.__indices2index_windows(indices, padding_length))
        return np.array(input_indices)

    def __indices2index_windows(self, seq_indices, padding_length):
        ext_indices = [self.dictionary[BOS]] * self.config.skip_left
        ext_indices.extend(seq_indices + [self.dictionary[EOS]] * self.config.skip_right)
        seq = []
        for index in range(self.config.skip_left, len(ext_indices) - self.config.skip_right):
            item = ext_indices[index - self.config.skip_left: index + self.config.skip_right + 1]
            seq.append(item)

        if len(seq) < padding_length:
            extra_items = [[BATCH_PAD] * self.config.concat_window_size] * (padding_length - len(seq))
            seq.extend(extra_items)
        return seq

    def viterbi(self, state, transition, init_transition=None):
        length = state.shape[1]
        path = np.ones([self.label_count, length], dtype=np.int32) * -1
        corr_path = np.zeros([length], dtype=np.int32)
        path_score = np.ones([self.label_count, length], dtype=np.float64) * (np.finfo('f').min / 2)

        if init_transition is not None:
            path_score[:, 0] = init_transition + state[:, 0]
        else:
            path_score[:, 0] = state[:, 0]

        for pos in range(1, length):
            for prev_label in range(self.label_count):
                for label in range(self.label_count):
                    score = path_score[prev_label][pos - 1] + transition[prev_label][label] + state[label][pos]
                    if score > path_score[label][pos]:
                        path[label][pos] = prev_label
                        path_score[label][pos] = score

        max_index = np.argmax(path_score[:, -1])
        corr_path[length - 1] = max_index
        for i in range(length - 1, 0, -1):
            max_index = path[max_index][i]
            corr_path[i - 1] = max_index

        labels = [self.reversed_label_mapping[label] for label in corr_path]
        return labels

    def _viterbi_training_stage(self, state, transition, init_transition, labels, padding_length):
        length = state.shape[1]
        path = np.ones([self.label_count, length], dtype=np.int32) * -1
        corr_path = np.zeros([padding_length], dtype=np.int32)
        path_score = np.ones([self.label_count, length], dtype=np.float64) * (np.finfo('f').min / 2)

        path_score[:, 0] = init_transition + state[:, 0]

        for i in range(self.label_count):
            if i != labels[0]:
                path_score[i, 0] += self.config.hinge_rate

        for pos in range(1, length):
            for prev_label in range(self.label_count):
                for cur_label in range(self.label_count):
                    score = path_score[prev_label][pos - 1] + transition[prev_label][cur_label] + state[cur_label][pos]
                    if labels[pos] != cur_label:
                        score += self.config.hinge_rate

                    if score > path_score[cur_label][pos]:
                        path[cur_label][pos] = prev_label
                        path_score[cur_label][pos] = score

        max_index = np.argmax(path_score[:, -1])
        corr_path[length - 1] = max_index
        for i in range(length - 1, 0, -1):
            max_index = path[max_index][i]
            corr_path[i - 1] = max_index

        return corr_path
