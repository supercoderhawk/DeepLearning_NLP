# -*- coding: utf-8 -*-
from dnlp.data_process.general.seq.process_cws import cws2norm
from dnlp.data_process.general.seq.norm2conll import json2conll
from dnlp.utils.utils import change_extname


def prepare_cws(src_filename, json_filename=None, dest_filename=None):
    if not dest_filename:
        dest_filename = change_extname(src_filename, '.conll')

    if not json_filename:
        json2conll(cws2norm(src_filename), dest_filename)
    else:
        json2conll(cws2norm(src_filename, json_filename), dest_filename)


def prepare_ner(src_filename, dest_filename=None):
    pass
