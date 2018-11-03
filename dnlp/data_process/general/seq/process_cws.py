# -*- coding: UTF-8 -*-
from dnlp.utils.utils import read_lines_in_file, write_json_file
from dnlp.utils.constant import TAG_BEGIN, TAG_INSIDE, TAG_END, TAG_SINGLE, REGEX_WHITESPACE


def cws2norm(src_filename, dest_filename=None, src_encoding='utf-8', ):
    """

    :param src_filename:
    :param dest_filename:
    :param src_encoding:
    :return:
    """
    lines = read_lines_in_file(src_filename, src_encoding)
    data = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        data.append(cws_line2norm(line))
    if dest_filename:
        write_json_file(data, dest_filename)

    return data


def cws_line2norm(line):
    entry = []
    offset = 0
    words = REGEX_WHITESPACE.split(line)
    for word in words:
        entry.extend(cws_word2norm(offset, word))
        offset += len(word)

    return entry


def cws_word2norm(offset, word):
    items = []
    if not word:
        raise Exception('word is empty')
    word_len = len(word)
    if word_len == 1:
        item = {'text': word, 'start': offset, 'end': offset + 1, 'label': TAG_SINGLE}
        items.append(item)
    else:
        start_item = {'text': word[0], 'start': offset, 'end': offset + 1, 'label': TAG_BEGIN}
        items.append(start_item)
        if word_len > 2:
            inside_items = []
            for idx, ch in enumerate(word[1:-1], 1):
                inside_item = {'text': ch, 'start': offset + idx, 'end': offset + idx + 1,
                               'label': TAG_INSIDE}
                inside_items.append(inside_item)
            items.extend(inside_items)
        end_item = {'text': word[-1], 'start': offset + word_len - 1, 'end': offset + word_len, 'label': TAG_END}
        items.append(end_item)
    return items
