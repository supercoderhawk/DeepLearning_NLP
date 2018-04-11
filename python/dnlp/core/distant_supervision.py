# -*- coding:utf-8 -*-
import pickle


def read_relation(path):
  relations = []
  with open(path, 'rb') as f:
    data = pickle.load(f)
    for entry in data:
      rel = (entry['words'][entry['primary']], entry['words'][entry['secondary']], entry['type'])
      relations.append(rel)
  return relations


def construct_kb(relations):
  knowledge_base = {}
  relations = set(relations)
  for p, s, t in relations:
    key = ':'.join((str(p), str(s)))
    if key not in knowledge_base:
      knowledge_base[key] = (p, s, t)
    else:
      print('fuck')
      print(knowledge_base[key])
      print(t)
  return knowledge_base


KB = construct_kb(read_relation('../dnlp/data/emr/emr_paper/emr_relation.rel'))


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
with open('../dnlp/data/emr/rel_names', 'rb') as f:
  REL_PAIR_NAMES_INIT = pickle.load(f)
  REL_PAIR_LIST = list([r.split(':') for r in REL_PAIR_NAMES_INIT])
REL_NAME_IDX = {}
relation_category_index = 0
REL_NAMES = {'PartOf': '部位', 'PropertyOf': '性质', 'DegreeOf': '程度', 'QualityValue': '定性值',
             'QuantityValue': '定量值', 'UnitOf': '单位', 'TimeOf': '持续时间', 'StartTime': '开始时间',
             'EndTime': '结束时间', 'Moment': '时间点', 'DateOf': '日期', 'ResultOf': '结果',
             'LocationOf': '地点', 'DiseaseTypeOf': '疾病分型分期', 'SpecOf': '规格', 'UsageOf': '用法',
             'DoseOf': '用量', 'FamilyOf': '家族成员', 'ModifierOf': '其他修饰词', 'UseMedicine': '用药',
             'LeadTo': '导致', 'Find': '发现', 'Confirm': '证实', 'Adopt': '采取', 'Take': '用药',
             'Limit': '限定', 'AlongWith': '伴随', 'Complement': '补足'}
for relation_category in REL_NAMES:
  REL_NAME_IDX[relation_category] = relation_category_index
  relation_category_index += 1

DICTIONARY = read_dictionary(DICT_PATH)


def isdigit(s):
  try:
    float(s)
    return True
  except Exception as e:
    return False


def extract_relaction(entity1, entity2, type1, type2):
  key = ':'.join((str(entity1), str(entity2)))
  if type2 == 'Quantity' and type1 + ':' + type2 in REL_PAIR_NAMES_INIT:
    return REL_NAME_IDX[REL_PAIR_NAMES_INIT[type1 + ':' + type2]]
  else:
    if key in KB:
      return KB[key][2]
    else:
      return -1


if __name__ == '__main__':
  pass
