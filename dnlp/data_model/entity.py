# -*- coding: utf-8 -*-

class Entity(object):
    def __init__(self, text,offset,entity_type):
        self.__text = text
        self.__entity_type = entity_type

    @property
    def text(self):
        return self.__text

    @property
    def entity_type(self):
        return self.__entity_type