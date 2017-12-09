#-*- coding: UTF-8 -*-
import os
import re
RE_SAPCE = re.compile('[ ]+')
class ProcessEMR(object):
  def __init__(self,base_folder:str):
    self.base_folder = base_folder
    self.files = self.get_files()
    self.read_annotations()

  def get_files(self):
    files = set()
    for l in os.listdir(self.base_folder):
      files.add(os.path.splitext(l)[0])
    return files

  def read_annotations(self):
    for file in self.files:
      filename = self.base_folder+file
      sentence_dict,periods = self.read_entities_in_single_file(filename+'.txt',filename+'.ann')
      sentence_words = self.read_cws_file(filename+'.cws',periods)


  def read_cws_file(self,cws_file,periods):
    with open(cws_file, encoding='utf-8') as f:
      words = f.read().replace('\n','').split(' ')
      word_lens = [len(w) for w in words]
      start_point = [0]+[sum(word_lens[:l]) for l in range(1,len(words)+1) ][:-1]
      word_index = {c:w for w,c in zip(start_point)}
      sentence_words = []
      for s,e in zip(periods[:-1],periods[1:]):
        sentence_words.append(words[word_index[s]:word_index[e]])
      return sentence_words


  def read_entities_in_single_file(self,raw_file,ann_file):
    data = {}
    with open(raw_file, encoding='utf-8') as r:
      sentence = r.read()
      rn_indices = [m.start() for m in re.finditer('\n', sentence)]
      spans_diff = {}
      if len(rn_indices):
        spans = zip([-1] + rn_indices, rn_indices + [len(sentence) + len(rn_indices)])
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
        period_spans[(s + 1, e + 1)] = s + 1

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
              if s <= start and end <= e:
                diff = spans_diff[(s, e)]
                start -= diff
                end -= diff
                flag = True
                break
            if not flag:
              print('a fucked world')
          if sentence[start:end] != text:
            print('=========')
            # print(end - start)
            # print(id)
            # print(ann_file)
            # print(sentence[start:end])
            # print(text)
            # print('fuck world')
            continue

          if len(period_spans):
            for s, e in period_spans:
              if s <= start and end <= e:
                new_sentence = sentence[s:e]
                if new_sentence not in sentence_dict:
                  print(ann_file)
                  print('fuck aa')
                new_diff = period_spans[(s, e)]
                start -= new_diff
                end -= new_diff
                if new_sentence[start:end] != text:
                  print('fuck')
                entity = {'id': id, 'start': start, 'length': end - start, 'text': text}
                entities = sentence_dict[new_sentence].get('entities')
                if entities is not None:
                  entities.append(entity)
                else:
                  sentence_dict[new_sentence]['entities'] = [entity]
                break

    return sentence_dict,periods