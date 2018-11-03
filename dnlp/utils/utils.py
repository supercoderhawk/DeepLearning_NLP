# -*- coding: UTF-8 -*-
"""utility functions"""
import json


def read_lines_in_file(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        return f.read().splitlines()


def write_lines_in_file(lines, filename, encoding='utf-8'):
    for index, line in enumerate(lines):
        if not isinstance(line, str):
            lines[index] = json.dumps(line)
    with open(filename, 'w', encoding=encoding) as f:
        f.write('\n'.join(lines))


def read_file(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        return f.read()


def write_file(string, filename, encoding='utf-8'):
    if not isinstance(string, str):
        raise Exception('input data is not string.')
    with open(filename, 'w', encoding=encoding) as f:
        f.write(string)


def read_json_file(filename, encoding='utf-8'):
    with open(filename, encoding=encoding) as f:
        return json.load(f)


def write_json_file(data, filename, encoding='utf-8'):
    with open(filename, 'w', encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False)


def read_conll_file(filename, encoding='utf-8', conll_delimiter=' '):
    data = read_file(filename, encoding)
    entries = data.split('\n\n')

    for entry in entries:
        yield list(zip(*[item.split(conll_delimiter) for item in entry.splitlines()]))


def read_conll_in_pandas():
    pass


def cws_label2word(words, labels, schema, check_label=False):
    if len(words) != len(labels):
        raise Exception('word and label length are not equal')
    if type(words) not in {str, list} or not isinstance(labels, list):
        raise Exception('word and label must be in list.')
    if not labels:
        return ''
    if not isinstance(schema, str):
        raise Exception('schema type error')
    if schema not in {'BILU', 'BIES'}:
        raise Exception('schema is not supported')

    if schema == 'BILU':
        start_label, inter_label, end_label, single_label = 'B', 'I', 'L', 'U'
    else:
        start_label, inter_label, end_label, single_label = 'B', 'I', 'E', 'S'

    tokens = []
    token = ''

    for word, label in zip(words, labels):
        if label == single_label:
            tokens.append(word)
        elif label == start_label:
            token = word
        elif label == inter_label:
            token += word
        else:
            tokens.append(token + word)

    return tokens


def check_cws_label(labels, start_label, inter_label, end_label, single_label):
    label_order = {start_label: [inter_label, end_label],
                   inter_label: [inter_label, end_label]}
    if len(labels) == 1:
        if labels[0] == single_label:
            return True
        else:
            return False
    for cur_label, next_label in zip(labels[:-1], labels[1:]):
        pass


def ner_label2entity(words, labels, schema):
    pass


def change_extname(filename, extname):
    """
    change the file extension. If the file doesn't have a extension, append designated extension to it.
    :param filename: source filename
    :param extname: extension name, If it doesn't start with dot, will add dot
    :return: changed filename
    """
    index = filename.rfind('.')

    if not extname.startswith('.'):
        extname = '.' + extname

    if index == -1:
        return filename + extname
    else:
        return filename[:index] + extname
