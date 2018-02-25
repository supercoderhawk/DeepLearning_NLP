# -*- coding:utf-8 -*-
from sklearn_crfsuite import CRF
# import pandas as pd
import joblib
from dnlp.crf.crf_feature import Feature
from dnlp.utils.utils import load_data_in_conll, labels2entity_texts
from dnlp.utils.evaluation import get_ner_statistics
from collections import Counter

ENTITY_CATEGORY = {'Sign': 'SN', 'Symptom': 'SYM', 'Part': 'PT', 'Property': 'PTY', 'Degree': 'DEG',
                   'Quality': 'QLY', 'Quantity': 'QNY', 'Unit': 'UNT', 'Time': 'T', 'Date': 'DT',
                   'Result': 'RES',
                   'Disease': 'DIS', 'DiseaseType': 'DIT', 'Examination': 'EXN', 'Location': 'LOC',
                   'Medicine': 'MED', 'Spec': 'SPEC', 'Usage': 'USG', 'Dose': 'DSE', 'Treatment': 'TRT',
                   'Family': 'FAM',
                   'Modifier': 'MOF'}
ENTITY_CATEGORY = dict(zip(ENTITY_CATEGORY.values(),ENTITY_CATEGORY.keys()))

class CrfModel(object):
  def __init__(self, base_folder='../models/crf/',model_name='crf_model.jl'):
    self.model_path = base_folder + model_name
    self.CRF = joblib.load(self.model_path)

  def inference(self, sentence, return_label=False):
    f = Feature(sentence).sentence2features()
    prediction = self.CRF.predict([f])[0]
    if not return_label:
      return labels2entity_texts(prediction, sentence,ENTITY_CATEGORY)
    else:
      return prediction

  def evaluate(self, src_file='../data/emr/emr_test_type.conll'):
    sentences, true_labels = load_data_in_conll(src_file)
    true_count = 0
    prec_count = 0
    recall_count = 0
    for sentence, true_label in zip(sentences, true_labels):
      pred_label = self.inference(sentence, True)
      t, p, r = get_ner_statistics(true_label, pred_label)
      true_count += t
      prec_count += p
      recall_count += r
    print(true_count / prec_count)
    print(true_count / recall_count)

  def analysis_features(self, top=8):
    trans_features = Counter(self.CRF.transition_features_).most_common(top)
    print("Top likely transitions:")
    for (label_from, label_to), weight in trans_features:
      print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


def feed_crf_trainer(filename, delimiter=' '):
  x, y = load_data_in_conll(filename, delimiter=delimiter, return_pd=True)
  x = x.apply(lambda s: Feature(''.join(s)).sentence2features())
  return x, y


def train_crf(x_train, y_train, c1=0.1, c2=0.1, algm='lbfgs', max_iter=100, all_trans=True):
  crf = CRF(
    algorithm=algm,
    c1=c1,
    c2=c2,
    max_iterations=max_iter,
    all_possible_transitions=all_trans
  )
  return crf.fit(x_train, y_train)


def training(src_filename='corpus.conll', dest_dir='../models/crf/', model_name='crf_model.jl'):
  x, y = feed_crf_trainer(src_filename)
  crf = train_crf(x, y)
  joblib.dump(crf, dest_dir + model_name)
  return crf


if __name__ == '__main__':
  data_path = '../data/emr/'
  training(data_path + 'emr_training_type.conll')
  crf = CrfModel()
  crf.evaluate()
  print(crf.inference('双肺呼吸音清'))
  print(crf.inference('多饮多尿多食'))
