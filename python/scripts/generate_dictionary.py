# -*- coding: UTF-8 -*-
import sys
import re
import argparse
from dnlp.utils.constant import BATCH_PAD, BATCH_PAD_VAL, UNK, UNK_VAL, STRT, STRT_VAL, END, END_VAL


def generate_chinese_dictionary(files):
  rx = re.compile('\s')
  characters = set()
  dictionary = {BATCH_PAD: BATCH_PAD_VAL, UNK: UNK_VAL, STRT: STRT_VAL, END: END_VAL}
  for file in files:
    with open(file, encoding='utf-8') as f:
      characters = characters.union(rx.sub('', f.read()))
  new_dict = dict(zip(characters, range(len(dictionary), len(dictionary) + len(characters))))
  dictionary.update(new_dict)
  return dictionary


def generate_english_dictionary(files):
  pass


def generate_dictionary_from_conll(files):
  dictionary = {BATCH_PAD: BATCH_PAD_VAL, UNK: UNK_VAL, STRT: STRT_VAL, END: END_VAL}
  characters = set()

  for file in files:
    with open(file, encoding='utf-8') as f:
      characters = characters.union([l.split(' ')[0] for l in f.read().splitlines()])
  if '' in characters:
    characters.remove('')

  for i, c in enumerate(characters, len(dictionary)):
    dictionary[c] = i
  return dictionary


def output_dictionary(dictionary, filename):
  with open(filename, 'w', encoding='utf-8') as f:
    for ch in dictionary:
      f.write(ch + ' ' + str(dictionary[ch]) + '\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--c', dest='c', action='store_true', default=False)
  parser.add_argument('--r', dest='r', action='store_true', default=False)
  parser.add_argument('files', nargs='+')
  parser.add_argument('--o', dest='o', nargs='?')
  args = parser.parse_args(sys.argv[1:])
  conll = args.c
  raw = args.r
  output = args.o
  files = args.files
  if not output:
    print('don\'t enter output dict file path')
  if not len(files):
    print('don\'t enter filenames')
    exit(1)
  if conll and raw:
    print('can\'t use two formats simultaneously.')
    exit(1)
  if not conll and not raw:
    print('don\'t enter dictionary mode')
  if raw:
    dictionary = generate_chinese_dictionary(files)
  else:
    dictionary = generate_dictionary_from_conll(files)

  output_dictionary(dictionary, output)
