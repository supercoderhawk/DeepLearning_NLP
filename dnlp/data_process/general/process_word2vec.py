# -*- coding: UTF-8 -*-
import random
import pickle
from dnlp.data_process.processor import Preprocessor
from dnlp.utils.constant import UNK


class EmbeddingPertrainProcess(Preprocessor):
    def __init__(self, base_folder: str, files: tuple, dict_path: str, skip_window: int,
                 output_name: str, mode: str = 'character', algorithm='skip_gram'):
        Preprocessor.__init__(self, base_folder=base_folder, dict_path=dict_path)
        self.skip_window = skip_window
        self.files = files
        self.output_name = output_name
        self.mode = mode
        self.sentences = self.preprocess()
        self.indices = self.map_to_indices()
        if algorithm == 'skip_gram':
            self.input, self.output = self.process_skip_gram()
        else:
            self.input, self.output = self.process_cbow()
        self.save_data()

    def preprocess(self):
        sentences = []
        for file in self.files:
            with open(self.base_folder + file, encoding='utf-8') as f:
                if self.mode == 'character':
                    sentences.extend(f.read().splitlines())
                else:
                    sentences.extend([l.split(' ') for l in f.read().splitlines()])
        return sentences

    def map_to_indices(self):
        indices = []

        for sentence in self.sentences:
            idx = []
            for c in sentence:
                if c in self.dictionary:
                    idx.append(self.dictionary[c])
                else:
                    idx.append(self.dictionary[UNK])
            indices.append(idx)
        return indices

    def process_cbow(self):
        input = []
        output = []
        for index in self.indices:
            if len(index) < 2 * self.skip_window + 1:
                continue
            for i in range(self.skip_window, len(index) - self.skip_window):
                input.append(index[i - self.skip_window:i] + index[i + 1:i + 1 + self.skip_window])
                output.append(index[i])
        return input, output

    def process_skip_gram(self):
        input = []
        output = []
        for index in self.indices:
            target = index[self.skip_window:-self.skip_window]
            input += [i for l in zip(*[target] * self.skip_window * 2) for i in l]
            target_index = range(self.skip_window, len(target) + self.skip_window)

            def shuffle(i):
                return random.sample(index[i - self.skip_window:i] + index[i + 1:i + self.skip_window + 1],
                                     2 * self.skip_window)

            output += [j for i in map(shuffle, target_index) for j in i]
            if len(input) != len(output):
                print(len(input) - len(output))

        return input, output

    def save_data(self):
        with open(self.base_folder + self.output_name, 'wb', ) as f:
            data = {'input': self.input, 'output': self.output, 'dictionary': self.dictionary}
            pickle.dump(data, f)
