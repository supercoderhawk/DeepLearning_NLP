# -*- coding:utf-8 -*-
import numpy as np
import pickle
import os
import json
import pprint
from collections import Counter, OrderedDict
from operator import itemgetter
from itertools import accumulate, permutations
from dnlp.config.sequence_labeling_config import DnnCrfConfig
from dnlp.crf.crf import CrfModel
from dnlp.core.dnn_crf import DnnCrf
from dnlp.core.re_cnn import RECNN
from dnlp.config.re_config import RECNNConfig
from dnlp.core.distant_supervision import extract_relaction
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
  REL_PAIR_NAMES_INIT = pickle.load(f)
  REL_PAIR_NAMES = dict(zip(REL_PAIR_NAMES_INIT.values(), REL_PAIR_NAMES_INIT.keys()))
  for rel_name in REL_PAIR_NAMES:
    REL_PAIR_NAMES[rel_name] = REL_PAIR_NAMES[rel_name].split(':')
REL_PAIR_LIST = list([r.split(':') for r in REL_PAIR_NAMES_INIT])
with open(BASE_FOLDER + 'type_dict.pickle', 'rb') as f:
  TYPE_DICT = pickle.load(f)
TYPE_DICT['3月'] = 'Time'
TYPE_DICT['12年'] = 'Time'
TYPE_DICT['多饮2天'] = 'Symptom'
TYPE_DICT['100/70'] = 'Quantity'
TYPE_DICT['18.314'] = 'Quantity'
TYPE_DICT['150/90'] = 'Quantity'
TYPE_DICT['29.297'] = 'Quantity'
TYPE_DICT['稍膨隆'] = 'Sign'
TYPE_DICT['左侧'] = 'Part'
TYPE_DICT['腰背部'] = 'Part'
TYPE_DICT['多处'] = 'Quality'
TYPE_DICT['境界'] = 'Part'
TYPE_DICT['0.5'] = 'Quantity'
TYPE_DICT['cm*1.0cm'] = 'Unit'
TYPE_DICT['发作心慌'] = 'Sign'
TYPE_DICT['湿性'] = 'Quality'
TYPE_DICT['左侧前'] = 'Part'
TYPE_DICT['血糖控制不佳'] = 'Sign'
TYPE_DICT['胸部'] = 'Part'
TYPE_DICT['第二至第四趾见创面缺损'] = 'Sign'
TYPE_DICT['粘稠'] = 'Quality'
TYPE_DICT['的分泌物'] = 'Sign'
TYPE_DICT['引流'] = 'Treatment'
TYPE_DICT['欠通畅'] = 'Symptom'
TYPE_DICT['稍降低'] = 'Sign'

REL_NAMES = {'PartOf': '部位', 'PropertyOf': '性质', 'DegreeOf': '程度', 'QualityValue': '定性值',
             'QuantityValue': '定量值', 'UnitOf': '单位', 'TimeOf': '持续时间', 'StartTime': '开始时间',
             'EndTime': '结束时间', 'Moment': '时间点', 'DateOf': '日期', 'ResultOf': '结果',
             'LocationOf': '地点', 'DiseaseTypeOf': '疾病分型分期', 'SpecOf': '规格', 'UsageOf': '用法',
             'DoseOf': '用量', 'FamilyOf': '家族成员', 'ModifierOf': '其他修饰词', 'UseMedicine': '用药',
             'LeadTo': '导致', 'Find': '发现', 'Confirm': '证实', 'Adopt': '采取', 'Take': '用药',
             'Limit': '限定', 'AlongWith': '伴随', 'Complement': '补足'}
REL_NAME_LIST = list(REL_NAMES.keys())
REL_NAME_IDX = {}
relation_category_index = 0
for relation_category in REL_NAMES:
  REL_NAME_IDX[relation_category] = relation_category_index
  relation_category_index += 1
REL_NAME_IDX = dict(zip(REL_NAME_IDX.values(), REL_NAME_IDX.keys()))
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


CRF = CrfModel('../dnlp/models/crf/')

relation_name_file = BASE_FOLDER + 'rel_names'
relation_pair_file = BASE_FOLDER + 'rel_pairs'
with open(relation_name_file, 'rb') as f:
  category_name = pickle.load(f)


# print(category_name)
def ner_crf(sentence):
  # res = CRF.inference(sentence)
  return CRF.inference(sentence)


def cws(sentence):
  config = DnnCrfConfig(skip_left=1, skip_right=1)
  model_path = '../dnlp/models/emr/cws-lstm-emr_cws-50.ckpt'
  dnncrf = DnnCrf(config=config, model_path=model_path, mode='predict', nn='lstm', task='cws', remark='emr_cws')
  return dnncrf.predict_ll(sentence)


def modify_cws(cws_res, entity, entity_start):
  lengths = [0] + list(accumulate([len(l) for l in cws_res]))
  entity_end = entity_start + len(entity)
  if entity_start not in lengths or entity_end not in lengths:
    prev_start = rear_start = -1
    for b, e in zip(lengths[:-1], lengths[1:]):
      if b < entity_start < e:
        prev_start = b
        # break
    if prev_start != -1:
      prev_index = lengths.index(prev_start)
      offset = entity_start - prev_start
      split_cws = [cws_res[prev_index][:offset], cws_res[prev_index][offset:]]
      cws_res = cws_res[:prev_index] + split_cws + cws_res[prev_index + 1:]
    lengths = [0] + list(accumulate([len(l) for l in cws_res]))
    for b, e in zip(lengths[:-1], lengths[1:]):
      if b < entity_end < e:
        rear_start = b
    if rear_start != -1:
      rear_index = lengths.index(rear_start)
      offset = entity_end - rear_start
      split_cws = [cws_res[rear_index][:offset], cws_res[rear_index][offset:]]
      cws_res = cws_res[:rear_index] + split_cws + cws_res[rear_index + 1:]

  lengths = [0] + list(accumulate([len(l) for l in cws_res]))
  end_index = lengths.index(entity_end)
  start_index = lengths.index(entity_start)
  if end_index - start_index > 1:
    cws_res = cws_res[:start_index] + [''.join(cws_res[start_index:end_index])] + cws_res[end_index:]

  return cws_res


def prepare_rel(sentence, batch_length=85):
  cws_res = cws(sentence)
  ner_res = ner(sentence)
  # ner_res = ner_crf(sentence)
  print(sentence)
  print(ner_res)
  print(cws_res)
  lengths = [0] + list(accumulate([len(l) for l in cws_res]))
  # print(cws_res)
  # print(lengths)
  ne_candidates = {}
  words = list(map(lambda w: DICTIONARY[w] if w in DICTIONARY else DICTIONARY[UNK], cws_res))
  if len(words) < batch_length:
    words += [DICTIONARY[BATCH_PAD]] * (batch_length - len(words))
  else:
    words = words[:batch_length]
  # print(ner_res)
  for ne, s in ner_res:
    if not (s in lengths and s + len(ne) in lengths and (lengths.index(s + len(ne)) - lengths.index(s) == 1)):
      cws_res = modify_cws(cws_res, ne, s)
      lengths = [0] + list(accumulate([len(l) for l in cws_res]))
    idx = lengths.index(s)
    ne_candidates[idx] = 1
    # print('fuck')
  # rel_candidates = list(permutations(ne_candidates, 2))
  rel_candidates_raw, entity_type_pairs_raw = get_candidate_rel(ne_candidates, cws_res)
  rel_candidates = []
  entity_type_pairs = []
  for entity_pair, type_pair in zip(rel_candidates_raw, entity_type_pairs_raw):
    # if type_pair in REL_PAIR_LIST:
    rel_candidates.append(entity_pair)
    entity_type_pairs.append(type_pair)
  primary, secondary = generate_rel(rel_candidates, batch_length)
  word_array = np.array([[words]] * len(rel_candidates))
  rel_count = len(rel_candidates)
  print(rel_count)
  return [words] * rel_count, primary, secondary, [cws_res] * rel_count, rel_candidates, entity_type_pairs


def get_candidate_rel(ne_candidate, words):
  # print(words)
  # print(ne_candidate.keys())
  comma_idx = [i for i, w in enumerate(words) if w == '，' or w == ',']

  # print(comma_idx, len(words))
  if not comma_idx:
    rel_candidates_entity_type = [(TYPE_DICT[words[p]], TYPE_DICT[words[s]]) for p, s in
                                  list(permutations(ne_candidate, 2))]
    return list(permutations(ne_candidate, 2)), rel_candidates_entity_type
  if comma_idx[-1] != len(words) - 1:
    comma_idx.append(len(words) - 1)
  spans = zip([0] + comma_idx[:-1], comma_idx)
  rel_candidates = []
  # print(ne_candidate)
  for s, e in spans:
    candidates = [c for c in ne_candidate if s <= c <= e]
    # candidates = [c for c in ne_candidate]
    # print(list(permutations(candidates, 2)))
    rel_candidates.extend(list(permutations(candidates, 2)))
  # print(rel_candidates)
  rel_candidates_entity_type = [[TYPE_DICT[words[p]], TYPE_DICT[words[s]]] for p, s in rel_candidates]
  filtered_rel_candidates = []
  filtered_rel_candidates_entity_type= []
  for rel,(ent1_type,ent2_type) in zip(rel_candidates,rel_candidates_entity_type):
    if ent1_type+':'+ent2_type in REL_PAIR_NAMES_INIT:
      filtered_rel_candidates.append(rel)
      filtered_rel_candidates_entity_type.append((ent1_type,ent2_type))
  return filtered_rel_candidates, filtered_rel_candidates_entity_type


def generate_rel(rel_candidates, batch_length):
  primary = []
  secondary = []
  for f, s in rel_candidates:
    primary.append(np.arange(batch_length) - f + batch_length - 1)
    secondary.append(np.arange(batch_length) - s + batch_length - 1)
  return primary, secondary


def rel_extract(sentences):
  words = []
  rel_pairs = []
  type_pairs = []
  sentence_words = []
  primary = []
  secondary = []
  for sentence in sentences:
    w, p, s, ww, pp, tp = prepare_rel(sentence)
    sentence_words.extend(w)
    primary.extend(p)
    secondary.extend(s)
    words.extend(ww)
    rel_pairs.extend(pp)
    type_pairs.extend(tp)
  sentence_words = np.asarray(sentence_words)
  primary = np.asarray(primary)
  secondary = np.asarray(secondary)
  config_two = RECNNConfig(window_size=(2,3,4))
  config_multi = RECNNConfig(window_size=(3, 4))
  model_path_two = '../dnlp/models/re_two/7-2_3_4_directed.ckpt'
  model_path_multi = '../dnlp/models/re_multi/13-3_4_directed.ckpt'
  recnn2 = RECNN(config=config_two, dict_path=DICT_PATH, mode='test', model_path=model_path_two, relation_count=2,
                 data_mode='test')
  recnn = RECNN(config=config_multi, dict_path=DICT_PATH, mode='test', model_path=model_path_multi, relation_count=28,
                data_mode='test')
  # print(primary.shape)
  # print(sentence_words.shape)

  two_res = recnn2.predict(sentence_words, primary, secondary)
  print(Counter(two_res))
  true_idx = [ii for ii, i in enumerate(two_res) if i]
  get_true_rel = itemgetter(*[ii for ii, i in enumerate(two_res) if i])
  true_words = get_true_rel(words)
  true_rel_pairs = get_true_rel(rel_pairs)
  true_entity_type_pairs = get_true_rel(type_pairs)
  true_sentence_words = get_true_rel(sentence_words)
  true_primary = get_true_rel(primary)
  true_secondary = get_true_rel(secondary)
  multi_res = recnn.predict(true_sentence_words, true_primary, true_secondary)
  true_idx_ds, true_types_ds = rel_extract_by_ds(sentence_words, rel_pairs, type_pairs)
  types_cnn = {i: r for i, r in zip(true_idx, multi_res)}
  types_ds = {i: r for i, r in zip(true_idx_ds, true_types_ds)}
  entity_type_cnn = {i: type_pairs[i] for i in true_idx}
  entity_type_ds = {i: type_pairs[i] for i in true_idx_ds}
  types_cnn.update(types_ds)
  entity_type_cnn.update(entity_type_ds)
  get_true_rel = itemgetter(*types_cnn.keys())
  true_words = get_true_rel(words)
  true_rel_pairs = get_true_rel(rel_pairs)
  return get_rel_result(true_words, true_rel_pairs, list(types_cnn.values()), list(entity_type_cnn.values()))
  # return get_rel_result(true_words,true_rel_pairs,multi_res,true_entity_type_pairs)


def rel_extract_by_ds(sent_words, rel_pairs, entity_type_pairs):
  true_idx = []
  true_types = []
  for idx, (words, (ent1_idx, ent2_idx), (ent_type1, ent_type2)) in enumerate(
    zip(sent_words, rel_pairs, entity_type_pairs)):
    ret = extract_relaction(words[ent1_idx], words[ent2_idx], ent_type1, ent_type2)
    if ret != -1:
      true_idx.append(idx)
      true_types.append(ret)

  return true_idx, true_types


def get_rel_result(words, rel_pairs, rel_types, type_pairs):
  relations = []
  result = {}
  print(len(rel_pairs))
  for sentence_words, (primary_idx, secondary_idx), rel_type, (primary_type, secondary_type) in zip(words, rel_pairs,
                                                                                                    rel_types,
                                                                                                    type_pairs):
    rel_type_name = REL_NAME_IDX[rel_type]
    # rel_type_name = REL_NAME_LIST[rel_type]
    if [primary_type, secondary_type] not in REL_PAIR_LIST:
      if [secondary_type, primary_type] in REL_PAIR_LIST:
        primary_type, secondary_type = secondary_type, primary_type
        primary_idx, secondary_idx = secondary_idx, primary_idx
      # else:
      #   continue
    primary = sentence_words[primary_idx]
    secondary = sentence_words[secondary_idx]
    # primary_type,secondary_type = REL_PAIR_NAMES[rel_type_name]
    # primary_type, secondary_type = REL_PAIR_NAMES[rel_type_name]
    rel_item = OrderedDict({'ent1': primary, 'ent2': secondary, 'ent1_type': primary_type,
                            'ent2_type': secondary_type,'rel_type': rel_type_name})
    primary_type = ENTITY_NAMES[primary_type]
    secondary_type = ENTITY_NAMES[secondary_type]
    rel = {'value': secondary, 'entity_type': primary_type, 'type': REL_NAMES[rel_type_name]}

    relations.append(rel_item)
    if not result.get(primary):
      result[primary] = [rel]
    else:
      result[primary].append(rel)
  # print(result)
  merged_result = {t: [] for t in set([rel[0]['entity_type'] for rel in result.values()])}
  for primary, value in result.items():
    res = {primary: {v['type']: v['value'] for v in value}}
    primary_type = value[0]['entity_type']
    merged_result[primary_type].append(res)
  # print(merged_result)
  # return merged_result
  return relations


def export():
  pass


def get_sentences(filename):
  with open('../dnlp/data/emr/emr_paper/test/' + filename, encoding='utf-8') as f:
    sentences = [l + '。' for l in f.read().split('。')]
    if sentences[-1] == '。':
      sentences = sentences[:-1]
    else:
      sentences[-1] = sentences[-1][:-1]
    return sentences


def evaluate_rel_result(pred_data, true_data):
  all_data = {f: (pred_data[f], true_data[f]) for f in pred_data.keys()}
  tp_count = 0
  true_count = 0
  pred_count = 0

  for pred_relations, true_relations in all_data.values():
    pred = {'-'.join(r.values()) for r in pred_relations}
    true = {'-'.join(r.values()) for r in true_relations}
    print('-------------')
    print(len(pred.intersection(true)))
    print(len(true))
    print(len(pred))
    tp_count += len(pred.intersection(true))
    true_count+=len(true)
    pred_count+=len(pred)

  print(tp_count/pred_count,tp_count/true_count)

def evaluate():
  files = set()
  # print(os.path.abspath())
  base_folder = '../dnlp/data/emr/emr_paper/test/'
  for l in os.listdir(base_folder):
    files.add(os.path.splitext(l)[0])
  pred_data = {}
  for f in files:
    filename = f + '.txt'
    print(filename)
    sentences = get_sentences(filename)
    res = rel_extract(sentences)
    pred_data[filename[:-4]] = res
    print(pred_data)
  with open('../dnlp/data/emr/emr_paper/emr_test_rel.pickle', 'rb') as f:
    true_data = pickle.load(f)
  evaluate_rel_result(pred_data, true_data)

if __name__ == '__main__':
  # filename = '994671_admission.txt'
  filename = '993769_admission.txt'
  # filename = '996716_admission.txt'
  evaluate()
  # sentences = get_sentences(filename)
  # res = rel_extract(sentences)
  # with open('../dnlp/data/emr/structured_example.json', 'w', encoding='utf-8') as f:
  #   f.write(pprint.pformat(res, width=100).replace('\'', '"'))
