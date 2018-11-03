# -*- coding: UTF-8 -*-
from dnlp.utils.utils import read_json_file, write_json_file, read_lines_in_file
from dnlp.utils.constant import REGEX_WHITESPACE


def ner2json(src_filename, dest_filename, src_encoding='utf-8'):
    data = read_json_file(src_filename, src_encoding)

    for entry in data:
        pass


def ner_entry2json(entry):
    text = entry['text']
    entities = entry['entities']
    words = entry['words']


def sort_entities():
    pass


def ner_whitespace2ner_format(src_filename, dest_filename, src_encoding='utf-8'):
    lines = read_lines_in_file(src_filename, src_encoding)
    data = []

    for line in lines:
        data.append(ner_line2ner_format(line))

    write_json_file(data, dest_filename)


def ner_line2ner_format(line, tokenizer=None, no_entity_tag='O', inside_delimiter='/'):
    """
    extract entities in such format:
    'Google/COMP is/O a/O good/O company/O ./O'
    :param line:
    :param tokenizer: tokenizer function to split words, if it is None, words will be split by default whitesapce
    :param no_entity_tag: The non-entity tag in line
    :param inside_delimiter: delimiter which is used to split word and tag.
    :return:
    """
    items = REGEX_WHITESPACE.split(line)
    tokens = []
    offset = 0
    entities = []
    text = REGEX_WHITESPACE.sub('', line)

    if tokenizer:
        tokens = tokenizer(text)

    for item in items:
        token, tag = item.split(inside_delimiter)
        if not tokenizer:
            tokens.append(token)
        if tag != no_entity_tag:
            entity = {'text': token, 'start': offset, 'end': offset + len(token), 'type': 'tag'}
            entities.append(entity)
        offset += len(token)

    return {'text': text, 'entities': entities, 'tokens': tokens}
