# -*- coding: UTF-8 -*-
"""
initialize the folders and data for training and testing
"""
import os
from dnlp.utils.constant import *
from dnlp.data_process.data_pipeline import prepare_cws


def init():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    if not os.path.exists(CWS_DIR):
        os.mkdir(CWS_DIR)
    if not os.path.exists(NER_DIR):
        os.mkdir(NER_DIR)
    if not os.path.exists(REL_DIR):
        os.mkdir(REL_DIR)


def prepare_cws_data():
    prepare_cws(CWS_PKU_TRAINING_SRC_FILE, CWS_PKU_TRAINING_JSON_FILE, CWS_PKU_TRAINING_FILE)
    prepare_cws(CWS_PKU_TEST_SRC_FILE, CWS_PKU_TEST_JSON_FILE, CWS_PKU_TEST_FILE)
    prepare_cws(CWS_MSR_TRAINING_SRC_FILE, CWS_MSR_TRAINING_JSON_FILE, CWS_MSR_TRAINING_FILE)
    prepare_cws(CWS_MSR_TEST_SRC_FILE, CWS_MSR_TEST_JSON_FILE, CWS_MSR_TEST_FILE)


if __name__ == '__main__':
    init()
    prepare_cws_data()
