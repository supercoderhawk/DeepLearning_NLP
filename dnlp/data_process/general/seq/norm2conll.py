# -*- coding: utf-8 -*-
import json
from dnlp.utils.utils import read_json_file, write_file
from dnlp.data_process.data_checker import check_sequence_json


def json2conll(data, dest_filename, conll_delimiter=' '):
    """
    transform json file to conll file
    :param data: json data
    :param dest_filename: destination conll file path
    :param conll_delimiter: delimiter between text and label in conll
    :return:
    """

    if not check_sequence_json(data=data):
        raise Exception('input data is not valid')

    conll_strs = []
    for entry in data:
        conll_str = '\n'.join([item['text'] + conll_delimiter + item['label'] for item in entry])
        conll_strs.append(conll_str)

    dest_data = '\n\n'.join(conll_strs)

    if dest_filename:
        write_file(dest_data, dest_filename)

    return dest_data