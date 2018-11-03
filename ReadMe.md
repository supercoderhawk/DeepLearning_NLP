# 基于深度学习的自然语言处理库

本项目是对[DeepNLP](https://github.com/supercoderhawk/DeepNLP)的重构，着重增强架构设计的合理性，提高代码的可读性，减少模块的耦合度，并增加一些新功能。

## 环境

* `python >= 3.5`
* `tensorflow >= 1.3.0`

## 项目结构
本项目的核心代码位于`python\dnlp`目录下

* `dnlp\data_process`: 数据预处理。
* `dnlp\seq_label`: 序列标注的模型代码，可用于分词、词性标注和实体识别。
* `dnlp\rel_extract`: 关系抽取的模型代码。
* `dnlp\joint_extract`: 实体和关系联合抽取的模型代码。
* `dnlp\runner`: 运行脚本

## 运行

1. 初始化数据

```bash
python python\scripts\init_datasets.py
```

2. 训练
```bash
python python\scripts\cws_ner.py -t
```

3. 使用
```bash
python python\scripts\cws_ner.py -p
```
## 参考论文

### 中文分词 && 命名实体识别
* [deep learning for chinese word segmentation and pos tagging](www.aclweb.org/anthology/D13-1061) （完全实现，文件[`dnn_crf.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/dnn_crf.py)）
* [Long Short-Term Memory Neural Networks for Chinese Word Segmentation](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP141.pdf) (完全实现，文件[`dnn_crf.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/dnn_crf.py))
* [Max-Margin Tensor Neural Network for Chinese Word Segmentation](www.aclweb.org/anthology/P14-1028) （待实现，文件[`mmtnn.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/mmtnn.py)）

## 实体关系抽取
* [relation extraction: perspective from convolutional neural networks](http://aclweb.org/anthology/W15-1506) （已实现，文件[`re_cnn.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/re_cnn.py)）


## ToDo-List

- [ ] 完善文档
- [ ] 增加更多算法的实现
- [ ] 支持pip
- [ ] 加入TensorBoard支持
- [ ] 支持TensorFlow Estimator和Save Model
- [ ] 增加对Java、C++的支持




