# -*- coding: UTF-8 -*-
from dnlp.utils.constant import LOSS_LOG_LIKEHOOD, NNCRF_DROPOUT_EMBEDDING


class NeuralNetworkCRFConfig(object):
    def __init__(self, *,
                 skip_left: int = 0,
                 skip_right: int = 0,
                 word_embed_size: int = 100,
                 hidden_layers: tuple = ({'type': 'lstm', 'units': 150},),
                 learning_rate: float = 0.05,
                 regularization_rate: float = 1e-4,
                 dropout_rate: float = 0.2,
                 dropout_position: str = NNCRF_DROPOUT_EMBEDDING,
                 batch_length: int = 300,
                 batch_size: int = 20,
                 hinge_rate: float = 0.2,
                 dict_path: str = '',
                 model_name: str = '',
                 training_filename: str = '',
                 label_schema: str = 'BIES',
                 loss_function_name: str = LOSS_LOG_LIKEHOOD):
        self.__skip_left = skip_left
        self.__skip_right = skip_right
        self.__word_embed_size = word_embed_size
        self.__hidden_layers = hidden_layers
        self.__learning_rate = learning_rate
        self.__regularization_rate = regularization_rate
        self.__dropout_rate = dropout_rate
        self.__dropout_position = dropout_position
        self.__batch_length = batch_length
        self.__batch_size = batch_size
        self.__hinge_rate = hinge_rate
        self.__dict_path = dict_path
        self.__model_name = model_name
        self.__training_filename = training_filename
        self.__label_schema = label_schema
        self.__loss_function_name = loss_function_name

    @property
    def skip_left(self):
        return self.__skip_left

    @property
    def skip_right(self):
        return self.__skip_right

    @property
    def word_embed_size(self):
        return self.__word_embed_size

    @property
    def hidden_layers(self):
        return self.__hidden_layers

    @property
    def learning_rate(self):
        return self.__learning_rate

    @property
    def regularization_rate(self):
        return self.__regularization_rate

    @property
    def dropout_rate(self):
        return self.__dropout_rate

    @property
    def dropout_position(self):
        return self.__dropout_position

    @property
    def batch_length(self):
        return self.__batch_length

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def hinge_rate(self):
        return self.__hinge_rate

    @property
    def dict_path(self):
        return self.__dict_path

    @property
    def model_name(self):
        return self.__model_name

    @property
    def training_filename(self):
        return self.__training_filename

    @property
    def concat_embed_size(self):
        return (self.__skip_right + self.__skip_left + 1) * self.__word_embed_size

    @property
    def concat_window_size(self):
        return self.__skip_left + self.__skip_right + 1

    @property
    def label_schema(self):
        return self.__label_schema

    @property
    def tag_count(self):
        return len(self.__label_schema)

    @property
    def loss_function_name(self):
        return self.__loss_function_name


class MMTNNConfig(object):
    def __init__(self, *, skip_left: int = 2,
                 skip_right: int = 2,
                 character_embed_size: int = 50,
                 label_embed_size: int = 50,
                 hidden_unit: int = 150,
                 learning_rate: float = 0.2,
                 lam: float = 10e-4,
                 dropout_rate: float = 0.4,
                 batch_length: int = 150,
                 batch_size: int = 20):
        self.__skip_left = skip_left
        self.__skip_right = skip_right
        self.__character_embed_size = character_embed_size
        self.__label_embed_size = label_embed_size
        self.__hidden_unit = hidden_unit
        self.__learning_rate = learning_rate
        self.__lam = lam
        self.__dropout_rate = dropout_rate
        self.__batch_length = batch_length
        self.__batch_size = batch_size

    @property
    def skip_left(self):
        return self.__skip_left

    @property
    def skip_right(self):
        return self.__skip_right

    @property
    def character_embed_size(self):
        return self.__character_embed_size

    @property
    def label_embed_size(self):
        return self.__label_embed_size

    @property
    def hidden_unit(self):
        return self.__hidden_unit

    @property
    def learning_rate(self):
        return self.__learning_rate

    @property
    def lam(self):
        return self.__lam

    @property
    def dropout_rate(self):
        return self.__dropout_rate

    @property
    def batch_length(self):
        return self.__batch_length

    @property
    def batch_size(self):
        return self.__batch_size
