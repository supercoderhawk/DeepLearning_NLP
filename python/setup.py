# -*- coding: UTF-8 -*-
from distutils.core import setup

setup(
  name='dnlp',
  version='0.0.1.5',
  packages=['dnlp'],
  url='https://github.com/supercodehawk/DeepLearning_NLP',
  license='MIT License',
  author='supercoderhawk',
  author_email='supercoderhawk@gmail.com',
  description='deep learning-based natural language processing lib',
  install_requires=[
  ],
  extra_require = {
    'tf':'tensorflow >= 1.3.0',
    'tf-gpu':'tensorflow_gpu >= 1.3.0'
  }
)
