# -*- coding: UTF-8 -*-
from dnlp.seq_label.nn_crf import NeuralNetworkCRF, NeuralNetworkCRFConfig
from dnlp.utils.constant import CWS_PKU_TRAINING_FILE, CWS_DICT_PATH, MODEL_DIR, LOSS_LOG_LIKEHOOD, LOSS_MAX_MARGIN
from dnlp.utils.utils import cws_label2word, read_conll_file
from dnlp.utils.evaluation import evaluate_cws
from dnlp.utils.constant import *


def cws_training():
    model_name = MODEL_DIR + 'cws_{0}.ckpt'
    # loss_function_name = LOSS_LOG_LIKEHOOD
    loss_function_name = LOSS_MAX_MARGIN
    config = NeuralNetworkCRFConfig(training_filename=CWS_PKU_TRAINING_FILE,
                                    dict_path=CWS_DICT_PATH,
                                    model_name=model_name,
                                    loss_function_name=loss_function_name,
                                    learning_rate=0.005,
                                    dropout_rate=0.2,
                                    regularization_rate=5e-4,
                                    skip_left=0,
                                    skip_right=0,
                                    word_embed_size=100,
                                    hidden_layers=({'type': 'lstm', 'units': 150},),
                                    dropout_position=NNCRF_DROPOUT_EMBEDDING,
                                    batch_length=300,
                                    batch_size=50,
                                    hinge_rate=0.2,
                                    label_schema='BIES')
    model = NeuralNetworkCRF(mode='fit', config=config, label2result=cws_label2word)
    model.fit()


def cws_predict():
    model_name = MODEL_DIR + 'cws_{}.ckpt'.format(1)
    config = NeuralNetworkCRFConfig(training_filename=CWS_PKU_TRAINING_FILE,
                                    dict_path=CWS_DICT_PATH,
                                    model_name=model_name,
                                    loss_function_name=LOSS_MAX_MARGIN)
    model = NeuralNetworkCRF(mode='inference', config=config, label2result=cws_label2word)
    ret1 = model.inference('我爱北京天安门。')
    ret2 = model.inference('小明来自南京师范大学')
    # ret2 = model.inference('小明是南京师范大学的学生')
    print(ret1)
    print(ret2)

    data = list(read_conll_file(CWS_PKU_TEST_FILE))
    evaluate_cws(model, data)


def ner_training():
    pass


if __name__ == '__main__':
    # cws_training()
    cws_predict()
