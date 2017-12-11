# -*- coding: UTF-8 -*-
import os
import re
import pickle
import random
from dnlp.utils.constant import UNK

RE_SAPCE = re.compile('[ ]+')


class ProcessEMR(object):
  def __init__(self, base_folder: str, dict_path: str = '', mode='train'):
    self.base_folder = base_folder
    self.data_folder = base_folder + 'emr_paper/'
    self.relation_name_file = base_folder + 'rel_names'
    self.relation_pair_file = base_folder + 'rel_pairs'
    with open(self.relation_name_file, 'rb') as f:
      self.category_name = pickle.load(f)
    # self.reversed_category_name = dict(zip(self.category_name.values(),self.category_name.keys()))
    self.mode = mode
    if self.mode =='train':
      self.window = 5
    else:
      self.window = 100
    self.dict_path = dict_path
    self.files = self.get_files()
    self.annotations = self.read_annotations()
    self.dictionary = self.read_dictionary()
    self.statistics()
    self.relation_categories = {'PartOf': '部位', 'PropertyOf': '性质', 'DegreeOf': '程度', 'QualityValue': '定性值',
                                'QuantityValue': '定量值', 'UnitOf': '单位', 'TimeOf': '持续时间', 'StartTime': '开始时间',
                                'EndTime': '结束时间', 'Moment': '时间点', 'DateOf': '日期', 'ResultOf': '结果',
                                'LocationOf': '地点', 'DiseaseTypeOf': '疾病分型分期', 'SpecOf': '规格', 'UsageOf': '用法',
                                'DoseOf': '用量', 'FamilyOf': '家族成员', 'ModifierOf': '其他修饰词', 'UseMedicine': '用药',
                                'LeadTo': '导致', 'Find': '发现', 'Confirm': '证实', 'Adopt': '采取', 'Take': '用药',
                                'Limit': '限定', 'AlongWith': '伴随', 'Complement': '补足'}
    self.relation_category_labels = {}
    relation_category_index = 0
    for relation_category in self.relation_categories:
      self.relation_category_labels[relation_category] = relation_category_index
      relation_category_index += 1
    print(len(self.relation_category_labels))
    with open(self.base_folder+'relation_index.pickle','wb') as f:
      pickle.dump(self.relation_category_labels,f)
    self.two_categories = self.generate_re_two_training_data()
    self.multi_categories = self.generate_re_mutli_training_data()
    self.save_data()

  def statistics(self):
    true_count = 0
    false_count = 0
    for annotation in self.annotations:
      true_count += len(annotation['true_relations'])
      false_count += len(annotation['false_relations'])
    all_count = true_count + false_count
    print(false_count / all_count)
    print(all_count)

  def generate_re_two_training_data(self):
    train_data = []
    for annotation in self.annotations:
      word_indices = self.map_to_indices(annotation['words'])
      for true_rel_name in annotation['true_relations']:
        true_rel = annotation['true_relations'][true_rel_name]
        train_data.append(
          {'words': word_indices, 'primary': true_rel['primary'], 'secondary': true_rel['secondary'], 'type': 1})
      for false_rel_name in annotation['false_relations']:
        false_rel = annotation['false_relations'][false_rel_name]
        train_data.append(
          {'words': word_indices, 'primary': false_rel['primary'], 'secondary': false_rel['secondary'], 'type': 0})
    random.shuffle(train_data)
    random.shuffle(train_data)
    random.shuffle(train_data)
    random.shuffle(train_data)
    random.shuffle(train_data)
    return train_data

  def generate_re_mutli_training_data(self):
    train_data = []
    for annotation in self.annotations:
      word_indices = self.map_to_indices(annotation['words'])
      for true_rel_name in annotation['true_relations']:
        true_rel = annotation['true_relations'][true_rel_name]
        train_data.append({'words': word_indices, 'primary': true_rel['primary'], 'secondary': true_rel['secondary'],
                           'type': self.relation_category_labels[true_rel['type']]})
    return train_data

  def map_to_indices(self, words):
    return list(map(lambda w: self.dictionary[w] if w in self.dictionary else self.dictionary[UNK], words))

  def save_data(self):
    with open(self.base_folder+self.mode+'_two.pickle','wb') as f:
      pickle.dump(self.two_categories,f)

    with open(self.base_folder+self.mode+'_multi.pickle','wb') as f:
      pickle.dump(self.multi_categories,f)



  def read_dictionary(self, reverse=False):
    dictionary = {}
    with open(self.dict_path, encoding='utf8') as d:
      items = d.readlines()
      for item in items:
        pair = item.split(' ')
        dictionary[pair[0]] = int(pair[1])
    if reverse:
      return dictionary, dict(zip(dictionary.values(), dictionary.keys()))
    else:
      return dictionary

  def get_files(self):
    files = set()
    print(os.path.abspath(self.data_folder))
    for l in os.listdir(self.data_folder + self.mode + '/'):
      files.add(os.path.splitext(l)[0])
    return files

  def read_annotations(self):
    all_sentences = []
    for file in self.files:
      filename = self.data_folder + self.mode + '/' + file
      sentence_dict, periods = self.read_entities_in_single_file(filename + '.txt', filename + '.ann')

      sentence_words = self.read_cws_file(self.data_folder + 'cws/' + file + '.cws', periods)
      sentences = [''.join(s) for s in sentence_words]
      for i, sentence in enumerate(sentences):
        if sentence not in sentence_dict:
          print(file)
          print(sentence)
        else:
          words = sentence_words[i]
          sentence_dict[sentence]['words'] = words
      remove_list = []
      for sentence in sentence_dict:
        # sentence = sentence_dict[sentence_text]
        if sentence_dict[sentence].get('entities') is None:
          remove_list.append(sentence)
      [sentence_dict.pop(r) for r in remove_list]
      for sentence_text in sentence_dict:
        sentence = sentence_dict[sentence_text]
        sentence['new_entities'] = {}
        for entity_id in sentence['entities']:
          entity_text = sentence['entities'][entity_id]['text']
          if not sentence.get('words'):
            print('aaa')
          sentence['entities'][entity_id]['index'] = sentence['words'].index(entity_text)
          entity = sentence_dict[sentence_text]['entities'][entity_id]
          if entity.get('index') is None:
            print('fuck your world')
          sentence['new_entities'][entity['index']] = entity

      data = self.read_relation_in_single_file(filename + '.ann', sentence_dict)
      all_sentences.extend(data.values())
    return all_sentences

  def read_relation_in_single_file(self, ann_file, data, directed=False):
    with open(ann_file, encoding='utf-8') as f:
      entries = map(lambda l: l.strip().split(' '), f.read().replace('\t', ' ').splitlines())
      for entry in entries:
        id = entry[0]
        if id.startswith('R'):
          primary = entry[2][entry[2].find(':') + 1:]
          # print(primary)
          secondary = entry[3][entry[3].find(':') + 1:]
          for sentence_text in data:
            sentence = data[sentence_text]
            entities = sentence['entities']
            # sentence['true_relations'] = {}
            if primary in entities and secondary in entities:
              rel = {'id': id, 'primary': entities[primary]['index'], 'secondary': entities[secondary]['index'],
                     'type': entry[1]}
              if sentence.get('true_realtions'):
                sentence['true_relations'][id] = rel
              else:
                sentence['true_relations'] = {id: rel}

      for sentence_text in data:
        sentence = data[sentence_text]
        if not sentence.get('true_relations'):
          print('sentence no relations')
          continue

        true_pairs = [(l['primary'], l['secondary']) for l in sentence['true_relations'].values()]
        comma_index = [i for i, w in enumerate(sentence['words']) if w in ('，', ',')]
        all_info = [(l['index'], l['type']) for l in sentence['entities'].values()]
        all_indices = sorted([l[0] for l in all_info])
        false_relations = {}
        for i, t in all_info:
          secondary_candidates = self.get_span(i, self.window, comma_index, all_indices, directed)
          candidates = [(f, s) for f, s in zip(len(secondary_candidates) * [i], secondary_candidates)]
          candidates = self.filter_entities(sentence['new_entities'], candidates)
          for f, s in [c for c in candidates if c not in true_pairs]:
            false_relations['-'.join([str(f), str(s)])] = {'primary': f, 'secondary': s}
        sentence['false_relations'] = false_relations
      remove_list = [s for s in data if not data[s].get('true_relations')]
      [data.pop(s) for s in remove_list]
    return data

  def filter_entities(self, entities_dict, candidate_pairs):
    type_pairs = [(entities_dict[p]['type'] + ':' + entities_dict[s]['type']) for p, s in candidate_pairs]
    return [candidate_pairs[i] for i, p in enumerate(type_pairs) if p in self.category_name]

  def get_span(self, index, window, comma_indices, true_indices, bidirectional=False):
    i = true_indices.index(index)
    spans = []
    right = [c for c in comma_indices if c > index]
    right_border = index + window
    if right:
      right_border = right[0] if right_border > right[0] else right_border
    spans.extend([c for c in true_indices[i + 1:] if c < right_border])
    if bidirectional:
      left = [c for c in comma_indices if c < index]
      left_border = 0 if index - window < 0 else index - window
      if left:
        left_border = left_border if left_border > left[-1] else left[-1]
      if i > 0:
        spans.extend([c for c in true_indices[:i - 1] if c > left_border])

    return spans

  def read_cws_file(self, cws_file, periods):
    if cws_file.find('995870_admission') != -1:
      print('aaaa')

    with open(cws_file, encoding='utf-8') as f:
      words = [w for w in RE_SAPCE.sub(' ', f.read().replace('\n', ' ')).split(' ') if w]
      word_lens = [len(w) for w in words]
      start_point = [0] + [sum(word_lens[:l]) for l in range(1, len(words) + 1)][:-1]
      word_index = {c: w for w, c in enumerate(start_point)}
      word_index[-1] = -1
      sentence_words = []
      a = start_point[-1]
      if a not in periods:
        periods.append(a)
      for s, e in zip([-1] + periods[:-1], periods):
        if not word_index.get(s) or not word_index.get(e):
          print(cws_file)
        sentence_words.append(words[word_index[s] + 1:word_index[e] + 1])

      return sentence_words

  def read_entities_in_single_file(self, raw_file, ann_file):
    data = {}
    with open(raw_file, encoding='utf-8') as r:
      sentence = r.read()
      rn_indices = [m.start() for m in re.finditer('\n', sentence)]
      spans_diff = {}
      if len(rn_indices):
        spans = zip([-2] + rn_indices, rn_indices + [len(sentence) + len(rn_indices) * 2])
        for i, (before, curr) in enumerate(spans):
          spans_diff[(before + 2, curr)] = i * 2
      raw_sentence = sentence
      sentence = sentence.replace('\n', '')

    # periods = [m.start() for m in re.finditer('。', sentence)]
    periods = []
    sentence_len = len(sentence)
    last = 0
    sentences = []
    for i, ch in enumerate(sentence):
      if ch == '。':
        if i < sentence_len - 1 and sentence[i + 1] == '”':
          continue
        else:
          periods.append(i)
          sentences.append(sentence[last:i + 1])
          last = i + 1
    if last != len(sentence):
      sentences.append(sentence[last:sentence_len])
    period_spans = {}
    sentence_dict = {k: {'text': k} for k in sentences}

    if len(periods):
      for s, e in zip([-1] + periods, periods + [len(sentence)]):
        period_spans[(s + 1, e)] = s + 1

    with open(ann_file, encoding='utf-8') as a:
      entries = map(lambda l: l.strip().split(' '), a.read().replace('\t', ' ').splitlines())

      for entry in entries:
        id = entry[0]
        if id.startswith('T'):
          start = int(entry[2])
          end = int(entry[3])
          text = entry[4]
          if len(rn_indices):
            flag = False
            for s, e in spans_diff:
              if s <= start and end < e:
                diff = spans_diff[(s, e)]
                start -= diff
                end -= diff
                flag = True
                break
            if not flag:
              print('not found')
              continue
          if sentence[start:end] != text:
            print('this is error')
            continue

          if len(period_spans):
            for s, e in period_spans:
              if s <= start and end <= e:
                new_sentence = sentence[s:e + 1]
                if new_sentence not in sentence_dict:
                  print(ann_file)
                  print('fuck aa')
                new_diff = period_spans[(s, e)]
                start -= new_diff
                end -= new_diff
                if new_sentence[start:end] != text:
                  print('fuck')
                entity = {'id': id, 'start': start, 'length': end - start, 'text': text, 'type': entry[1]}
                entities = sentence_dict[new_sentence].get('entities')
                if entities:
                  entities[id] = entity
                else:
                  sentence_dict[new_sentence]['entities'] = {id: entity}
                break
          else:
            entity = {'id': id, 'start': start, 'length': end - start, 'text': text, 'type': entry[1]}
            entities = sentence_dict[sentence].get('entities')
            entities[id] = entity

    return sentence_dict, periods
