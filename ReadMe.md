# 基于深度学习的自然语言处理库

本项目是对[DeepNLP](https://github.com/supercoderhawk/DeepNLP)的重构，着重增强架构设计的合理性，提高代码的可读性，减少模块的耦合度，并增加一些新功能。

## 环境

* `python >= 3.5`
* `tensorflow >= 1.3.0`

## 项目结构
本项目的核心代码位于`python\dnlp`目录下

```bash
python/dnlp
│  cws.py   # 分词
│  ner.py   # 命名实体识别
│  rel_extract.py # 关系抽取
│  __init__.py
│
├─config
│     config.py  # 配置项
│     __init__.py
│  
├─core  # 核心功能模块
│  │  dnn_crf.py    # 基于dnn-crf的序列标注
│  │  dnn_crf_base.py # 基于dnn-crf的序列标注的基类
│  │  mmtnn.py      # max-margin tensor nural network模型
│  │  re_cnn.py     # 基于cnn的关系抽取
│  │  __init__.py
│  
├─data_process  # 训练和测试数据的预处理
│     processor.py  # 基类
│     process_cws.py  # 对分词的预处理 
│     process_emr.py 
│     process_ner.py  # 对命名实体识别的预处理
│     process_pos.py  # 对词性标注的预处理
│     __init__.py
│  
│
├─models  # 保存训练后的模型
│
├─tests  # 单元测试
├─utils  # 公用函数
      constant.py  # 一些常量
      __init__.py
  
```

* `python\init_datasets.py`：初始化训练数据
* `python\runner\cws_ner.py`：进行分词和命名实体识别的训练和使用

## 运行

1. 初始化数据

```bash
python python\init_datasets.py
```

2. 训练
```bash
python dnlp\runner\cws_new.py -t
```

3. 使用
```bash
python dnlp\runner\cws_new.py -p
```
## 参考论文

### 中文分词 && 命名实体识别
* [deep learning for chinese word segmentation and pos tagging](www.aclweb.org/anthology/D13-1061) （完全实现，文件[`dnn_crf.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/dnn_crf.py)）
* [Long Short-Term Memory Neural Networks for Chinese Word Segmentation](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP141.pdf) (完全实现，文件[`dnn_crf.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/dnn_crf.py))
* [Max-Margin Tensor Neural Network for Chinese Word Segmentation](www.aclweb.org/anthology/P14-1028) （待实现，文件[`mmtnn.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/mmtnn.py)）

## 实体关系抽取
* [relation extraction: perspective from convolutional neural networks](http://aclweb.org/anthology/W15-1506) （待实现，文件[`re_cnn.py`](https://github.com/supercoderhawk/DeepLearning_NLP/blob/master/python/dnlp/core/re_cnn.py)）


## ToDo-List

- [ ] 增加更多算法的实现
- [ ] 支持pip
- [ ] 增加对Java、C++的支持




