# -*- coding: utf-8 -*-
"""
The functions in this module is to check whether data is valid.
The checked data is used to train, validate and test.
"""
from dnlp.utils.utils import read_json_file


def check_sequence_json(*, src_filename=None, data=None):
    """
    check whether the input data comply with json format (not check content).
    The sequence json formant is a abstracted format for storing sequence labeling task data.
    The data of sequence json is nest list, and every entry is list. Every item in entry dict.
    Every item has four keys: text, start, end and label
    For Example:
    [[{'text':'Google','start':0, 'end':6, 'label':'B-COMPANY'}]]
    src_filename and data must assign one and just one.
    :param src_filename: source file path
    :param data: source data
    :return: True if format is correct, else return False
    """
    if not src_filename and not data:
        raise Exception('not assign src_file or data')
    elif src_filename and data:
        raise Exception('assign src_file and data at same time')

    if src_filename:
        data = read_json_file(src_filename)

    if not isinstance(data, list):
        raise Exception('input data is not list')
    for entry in data:
        if not isinstance(entry, list):
            raise Exception('item in input data is not list')
        if not check_sequence_json_entry(entry):
            return False

    return True


def check_sequence_json_entry(data):
    for item in data:
        if not isinstance(item, dict) or len(item) != 4:
            return False
        is_correct = 'text' in item and 'start' in item and 'end' in item and 'label' in item

        if not is_correct:
            return False
        if not isinstance(item['start'], int) or not isinstance(item['end'], int):
            return False
        if not isinstance(item['text'], str) or not isinstance(item['label'], str):
            return False
        if item['start'] < 0 or item['end'] < 0 or item['start'] >= item['end']:
            return False

    return True
