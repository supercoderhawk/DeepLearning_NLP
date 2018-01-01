# -*- coding:utf-8 -*-
import numpy as np
import pickle
from itertools import accumulate, permutations
from dnlp.config.sequence_labeling_config import DnnCrfConfig
from dnlp.core.dnn_crf import DnnCrf
from dnlp.core.re_cnn import RECNN
from dnlp.config.re_config import RECNNConfig
from dnlp.utils.constant import UNK, BATCH_PAD


def read_dictionary(dict_path: str, reverse=False):
  dictionary = {}
  with open(dict_path, encoding='utf8') as d:
    items = d.readlines()
    for item in items:
      pair = item.split(' ')
      dictionary[pair[0]] = int(pair[1])
  if reverse:
    return dictionary, dict(zip(dictionary.values(), dictionary.keys()))
  else:
    return dictionary


BASE_FOLDER = '../dnlp/data/emr/'
DICT_PATH = BASE_FOLDER + 'emr_merged_word_dict.utf8'
DICTIONARY = read_dictionary(DICT_PATH)
with open(BASE_FOLDER + 'rel_names', 'rb') as f:
  REL_PAIR_NAMES = pickle.load(f)
  REL_PAIR_NAMES = dict(zip(REL_PAIR_NAMES.values(), REL_PAIR_NAMES.keys()))
  for rel_name in REL_PAIR_NAMES:
    REL_PAIR_NAMES[rel_name] = REL_PAIR_NAMES[rel_name].split(':')

REL_NAMES = {'PartOf': '部位', 'PropertyOf': '性质', 'DegreeOf': '程度', 'QualityValue': '定性值',
             'QuantityValue': '定量值', 'UnitOf': '单位', 'TimeOf': '持续时间', 'StartTime': '开始时间',
             'EndTime': '结束时间', 'Moment': '时间点', 'DateOf': '日期', 'ResultOf': '结果',
             'LocationOf': '地点', 'DiseaseTypeOf': '疾病分型分期', 'SpecOf': '规格', 'UsageOf': '用法',
             'DoseOf': '用量', 'FamilyOf': '家族成员', 'ModifierOf': '其他修饰词', 'UseMedicine': '用药',
             'LeadTo': '导致', 'Find': '发现', 'Confirm': '证实', 'Adopt': '采取', 'Take': '用药',
             'Limit': '限定', 'AlongWith': '伴随', 'Complement': '补足'}
REL_NAME_LIST = list(REL_NAMES.keys())
ENTITY_NAMES = {'Sign': '体征', 'Symptom': '症状', 'Part': '部位', 'Property': '属性', 'Degree': '程度',
                'Quality': '定性值', 'Quantity': '定量值', 'Unit': '单位', 'Time': '时间', 'Date': '日期',
                'Result': '结果',
                'Disease': '疾病', 'DiseaseType': '疾病类型', 'Examination': '检查', 'Location': '地址',
                'Medicine': '药物', 'Spec': '规格', 'Usage': '用法', 'Dose': '用量', 'Treatment': '治疗',
                'Family': '家族史',
                'Modifier': '修饰词'}


def ner(sentence):
  data_path = ''
  model_path = '../dnlp/models/emr/ner-lstm-50.ckpt'
  config = DnnCrfConfig(skip_left=1, skip_right=1)
  lstmcrf = DnnCrf(config=config, task='ner', model_path=model_path, mode='predict', data_path=data_path, nn='lstm',
                   remark='lstm')
  return lstmcrf.predict_ll(sentence)


def cws(sentence):
  config = DnnCrfConfig(skip_left=1, skip_right=1)
  model_path = '../dnlp/models/emr/cws-lstm-emr_cws-50.ckpt'
  dnncrf = DnnCrf(config=config, model_path=model_path, mode='predict', nn='lstm', task='cws', remark='emr_cws')
  return dnncrf.predict_ll(sentence)


def prepare_rel(sentence, batch_length=85):
  cws_res = cws(sentence)
  ner_res = ner(sentence)
  lengths = list(accumulate([len(l) for l in cws_res]))
  ne_candidates = []
  words = list(map(lambda w: DICTIONARY[w] if w in DICTIONARY else DICTIONARY[UNK], cws_res))
  if len(words) < batch_length:
    words += [DICTIONARY[BATCH_PAD]] * (batch_length - len(words))
  else:
    words = words[:batch_length]
  for ne, s in ner_res:
    idx = cws_res.index(ne)
    if idx != -1:
      ne_candidates.append(idx)
    else:
      print('fuck')
  rel_candidates = list(permutations(ne_candidates, 2))
  primary, secondary = generate_rel(rel_candidates, batch_length)
  word_array = np.array([[words]] * len(rel_candidates))
  rel_count = len(rel_candidates)
  return np.array([words] * rel_count), primary, secondary, [cws_res] * rel_count, rel_candidates


def generate_rel(rel_candidates, batch_length):
  primary = []
  secondary = []
  for f, s in rel_candidates:
    primary.append(np.arange(batch_length) - f + batch_length - 1)
    secondary.append(np.arange(batch_length) - s + batch_length - 1)
  return np.array(primary), np.array(secondary)


def rel_extract(sentences):
  words = []
  rel_pairs = []
  sentence_words = []
  primary = []
  secondary = []
  for sentence in sentences:
    w, p, s, ww, pp = prepare_rel(sentence)
    sentence_words.extend(w)
    primary.extend(p)
    secondary.extend(s)
    words.extend(ww)
    rel_pairs.extend(pp)
  config_two = RECNNConfig(window_size=(2,3,4))
  config_mutli = RECNNConfig(window_size=(2, 3, 4))
  model_path_two = '../dnlp/models/re_two/50-2_3_4_directed.ckpt'
  model_path_multi = '../dnlp/models/re_multi/50-2_3_4_directed.ckpt'
  recnn2 = RECNN(config=config_two, dict_path=DICT_PATH, mode='test', model_path=model_path_two, relation_count=2,data_mode='test')
  recnn = RECNN(config=config_two, dict_path=DICT_PATH, mode='test', model_path=model_path_multi, relation_count=28,data_mode='test')
  two_res = recnn2.predict(sentence_words, primary, secondary)
  true_words = [words[i] for i in two_res if i]
  true_rel_pairs = [rel_pairs[i] for i in two_res if i ]
  true_sentence_words = [sentence_words[i] for i in two_res if i]
  true_primary = [primary[i] for i in two_res if i]
  true_secondary = [secondary[i] for i in two_res if i]
  multi_res = recnn.predict(true_sentence_words, true_primary, true_secondary)
  get_rel_result(true_words,true_rel_pairs,multi_res)

def get_rel_result(words, rel_pairs,rel_types):
  result = {}
  for sentence_words, (primary_idx,secondary_idx),rel_type in zip(words,rel_pairs,rel_types):
    rel_type_name = REL_NAME_LIST[rel_type]
    primary = sentence_words[primary_idx]
    secondary = sentence_words[secondary_idx]
    primary_type,secondary_type = REL_PAIR_NAMES[rel_type_name]
    primary_type = ENTITY_NAMES[primary_type]
    secondary_type = ENTITY_NAMES[secondary_type]
    # result[]


def export():
  pass

def get_sentences(filename):
  with open('../dnlp/data/emr/emr_paper/train/' + filename, encoding='utf-8') as f:
    sentences = [l + '。' for l in f.read().split('。')]
    if sentences[-1] == '。':
      sentences = sentences[:-1]
    else:
      sentences[-1] = sentences[-1][:-1]
    return sentences


if __name__ == '__main__':
  sentences = get_sentences('996716_admission.txt')
  rel_extract(sentences)
