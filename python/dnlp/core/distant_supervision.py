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
def extract_relaction(entity1, entity2):
  key = ':'.join((str(entity1),str(entity2)))
  if key in KB:
    return KB[key][2]
  else:
    return -1


if __name__ == '__main__':
  pass
