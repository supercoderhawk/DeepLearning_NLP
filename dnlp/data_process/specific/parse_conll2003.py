# -*- coding: utf-8 -*-
from dnlp.utils.utils import read_file


def parse_conll2003(src_filename, dest_filename=None):
    text = read_file(src_filename)
    sents = text.split('\n\n')[1:]
    sent_words = []
    sent_pos_tags = []
    sent_chunk = []
    sent_labels = []

    for sent in sents:
        words, pos, chunk, label = parse_sentence(sent)
        sent_words.append(words)
        sent_pos_tags.append(pos)
        sent_chunk.append(chunk)
        sent_labels.append(label)


def parse_sentence(sent_text):
    words = []
    pos_tags = []
    chunks = []
    labels = []

    for item in sent_text.splitlines():
        word, pos, chunk, label = item.split()
        words.append(word)
        pos_tags.append(pos)
        chunks.append(chunk)
        labels.append(label)

    return words, pos_tags, chunks, labels
