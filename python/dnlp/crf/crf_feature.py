# -*- coding:utf-8 -*-
FEATURE_DELIMITER = ' '
CHARS = ['有','无','多','少','上','下','左','右','肝','肾','脾','胃','肋','心','肺','病','疼','痛']

class Feature(object):
  def __init__(self, sentence):
    self.words = list(sentence)
    self.words_count = len(self.words)

  def sentence2features(self):
    return [self.__word2features(i) for i in range(self.words_count)]

  def __word2features(self, idx):
    features = self.__get_common_features(idx)
    extensive_features = {
      'CHARS':self.words[idx] in CHARS,
    }
    features.update(extensive_features)

    return features

  def __get_common_features(self, idx):
    features = {
      'UNI:0': self.words[idx]
    }
    if idx >= 1:
      features.update({
        'UNI:-1': self.words[idx - 1],
        'UNI:-1/0': FEATURE_DELIMITER.join(self.words[idx - 1:idx + 1])
      })

      if idx < self.words_count - 1:
        features.update({
          'UNI:1': self.words[idx + 1],
          'UNI:0/1': FEATURE_DELIMITER.join(self.words[idx:idx + 2]),
          'UNI:-1/0/1': FEATURE_DELIMITER.join(self.words[idx - 1:idx + 2]),
        })

      if idx >= 2:
        features.update({
          'UNI:-2': self.words[idx - 2],
          'UNI:-2/-1/0': FEATURE_DELIMITER.join(self.words[idx - 2:idx + 1])
        })
        if idx < self.words_count - 2:
          features.update({
            'UNI:2': self.words[idx + 2],
            'UNI:0/1/2': FEATURE_DELIMITER.join(self.words[idx:idx + 3])
          })

    return features
