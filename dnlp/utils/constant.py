# -*- coding: UTF-8 -*-
import os
import re

# special characters in dictionary
BATCH_PAD = '<BATCH_PAD>'
BATCH_PAD_VAL = 0
UNK = '<UNK>'
UNK_VAL = 1
BOS = '<BOS>'
BOS_VAL = 2
EOS = '<EOS>'
EOS_VAL = 3

# tags
TAG_PAD = 'P'
TAG_BEGIN = 'B'
TAG_INSIDE = 'I'
TAG_END = 'E'
TAG_SINGLE = 'S'
TAG_OTHER = 'O'

# tag mapping table
CWS_TAGS = (TAG_BEGIN, TAG_INSIDE, TAG_END, TAG_SINGLE, TAG_PAD)
NER_TAGS = (TAG_BEGIN, TAG_INSIDE, TAG_OTHER, TAG_PAD)

# path related constant
BASE_DIR = os.path.realpath(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..')) + '/'
DATASET_DIR = BASE_DIR + 'datasets/'
DATA_DIR = BASE_DIR + 'data/'
MODEL_DIR = DATA_DIR + 'models/'
LOG_DIR = DATA_DIR + 'logs/'
CWS_DIR = DATA_DIR + 'cws/'
NER_DIR = DATA_DIR + 'ner/'
REL_DIR = DATA_DIR + 'rel/'

CWS_PKU_TRAINING_SRC_FILE = DATASET_DIR + 'pku_training.utf8'
CWS_PKU_TEST_SRC_FILE = DATASET_DIR + 'pku_test.utf8'
CWS_MSR_TRAINING_SRC_FILE = DATASET_DIR + 'msr_training.utf8'
CWS_MSR_TEST_SRC_FILE = DATASET_DIR + 'msr_test.utf8'

CWS_PKU_TRAINING_JSON_FILE = CWS_DIR + 'pku_training.json'
CWS_PKU_TEST_JSON_FILE = CWS_DIR + 'pku_test.json'
CWS_MSR_TRAINING_JSON_FILE = CWS_DIR + 'msr_training.json'
CWS_MSR_TEST_JSON_FILE = CWS_DIR + 'msr_test.json'

CWS_PKU_TRAINING_FILE = CWS_DIR + 'pku_training.conll'
CWS_PKU_TEST_FILE = CWS_DIR + 'pku_test.conll'
CWS_MSR_TRAINING_FILE = CWS_DIR + 'msr_training.conll'
CWS_MSR_TEST_FILE = CWS_DIR + 'msr_test.conll'

CWS_DICT_PATH = CWS_DIR + 'dict.utf8'

# regex
REGEX_WHITESPACE = re.compile('[ ]+')

# training and inference related constant
MODE_FIT = 'fit'
MODE_INFERENCE = 'inference'
LOSS_LOG_LIKEHOOD = 'log likehood'
LOSS_MAX_MARGIN = 'max margin'
LOSS_CROSS_ENTROPY_SOFTMAX = 'cross entroy softmax'
# NNCRF specific
NNCRF_DROPOUT_EMBEDDING = 'embedding'
NNCRF_DROPOUT_HIDDEN = 'hidden'
