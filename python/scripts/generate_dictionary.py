# -*- coding: UTF-8 -*-
import sys
import os
import re
import argparse
from dnlp.utils.constant import BATCH_PAD, BATCH_PAD_VAL, UNK, UNK_VAL, STRT, STRT_VAL, END, END_VAL
RE_SAPCE = re.compile('[ ]+')

def generate_chinese_character_dictionary(files):
  rx = re.compile('\s')
  characters = set()
  dictionary = {BATCH_PAD: BATCH_PAD_VAL, UNK: UNK_VAL, STRT: STRT_VAL, END: END_VAL}
  for file in files:
    with open(file, encoding='utf-8') as f:
      characters = characters.union(rx.sub('', f.read()))
  new_dict = dict(zip(characters, range(len(dictionary), len(dictionary) + len(characters))))
  dictionary.update(new_dict)
  return dictionary

def generate_word_dictionary(files):
  dictionary = {BATCH_PAD: BATCH_PAD_VAL, UNK: UNK_VAL, STRT: STRT_VAL, END: END_VAL}
  words = set()
  for file in files:
    with open(file, encoding='utf-8') as f:
      lines = RE_SAPCE.sub(' ',f.read()).splitlines()
      words.update(*[l.split(' ') for l in lines])
  if '' in words:
    words.remove('')
  new_dict = dict(zip(words, range(len(dictionary), len(dictionary) + len(words))))
  dictionary.update(new_dict)
  return dictionary



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
  parser.add_argument('--c', dest='conll', action='store_true', default=False)
  parser.add_argument('--r', dest='raw', action='store_true', default=False)
  parser.add_argument('--w',dest='word',action='store_true',default=False)
  parser.add_argument('files', nargs='*')
  parser.add_argument('--folder', dest='folder',type=str,nargs='?')
  parser.add_argument('--o', dest='o', nargs='?')
  args = parser.parse_args(sys.argv[1:])
  conll = args.conll
  raw = args.raw
  output = args.o
  files = args.files
  if not output:
    print('don\'t enter output dict file path')
  if not len(files) and not args.folder:
    print('don\'t enter filenames')
    exit(1)
  if conll and raw:
    print('can\'t use two formats simultaneously.')
    exit(1)
  if not conll and not raw and not args.word:
    print('don\'t enter dictionary mode')
  if raw:
    dictionary = generate_chinese_character_dictionary(files)
  elif args.word:
    if args.folder:
      filenames = [args.folder+fn for fn in os.listdir(args.folder)]
      dictionary = generate_word_dictionary(filenames)
    else:
      dictionary = generate_word_dictionary(files)
  else:
    dictionary = generate_dictionary_from_conll(files)

  output_dictionary(dictionary, output)
