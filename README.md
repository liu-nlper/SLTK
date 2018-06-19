# SLTK - Sequence Labeling Toolkit

序列化标注工具，基于PyTorch实现BLSTM-CNN-CRF模型，CoNLL 2003 English NER测试集F1值为91.10%（word and char feature）。

## 1. 快速开始

### 1.1 安装依赖项

    $ sudo pip3 install -r requirements.txt --upgrade  # for all user
    $ pip3 install -r requirements.txt --upgrade --user  # for current user

### 1.2 预处理&训练

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml -p --train

### 1.3 训练

若已经完成了预处理，则可直接进行训练:

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml --train

### 1.4 测试

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/word.yml --test


## 2. 配置文件说明

修改配置文件需遵循`yaml`语法格式。

### 2.1 训练|开发|测试数据

数据为`conllu`格式，每列之间用制表符或空格分隔，句子之间用空行分隔，标签在最后一列(若有标签)。

修改配置文件中`data_params`下的`path_train`，`path_dev`和`path_test`参数。其中，若`path_dev`为空，则在训练时会按照`model_params.dev_size`参数，将训练集划分一部分作为开发集。

### 2.2 特征

若训练数据包含多列特征，则可通过修改配置文件中的`data_params.feature_cols`指定使用其中某几列特征，`data_params.feature_names`为特征的别名，需和`data_params.feature_cols`等长。

`data_params.alphabet_params.min_counts`: 在构建特征的词汇表时，该参数用于过滤频次小于指定值的特征。

`model_params.embed_sizes`: 指定特征的维度，若提供预训练特征向量，则以预训练向量维度为准。

`model_params.require_grads`: 设定特征的embedding table是否需要更新。

`model_params.use_char`: 是否使用char level的特征。

### 2.3 预训练特征向量

`data_params.path_pretrain`: 指定预训练的特征向量，该参数中元素格式需要和`data_params.feature_names`中的顺序一致(可设为null)。

### 2.4 其他特征

`word_norm`: 是否对单词中的数字进行处理（仅将数字转换为0）；

`max_len_limit`: batch的长度限制。训练时，一个批量的长度是由该批量中最长的句子决定的，若最大句子长度超出此限制，则该批量长度被强制设为该值；

`all_in_memory`: 预处理之后，数据被存放在hdf5格式文件中，该数据对象默认存储在磁盘中，根据索引值实时进行加载；若需要加快数据读取速度，可将该值设为`true`(适用于小数据量)。


## 3. 性能

下表列出了在CoNLL 2003 NER测试集的性能，特征和参数设置与Ma等（2016）一致。

**表.** 实验结果

| 模型 | % P | % R | % F1 |
| ------------ | ------------ | ------------ | ------------ |
| Lample et al. (2016) | -| - | 90.94 |
| Ma et al. (2016) | 91.35 | 91.06 | 91.21 |
| BGRU | 85.50 | 85.89 | 85.69 |
| BLSTM | 88.05 | 87.19 | 87.62 |
| BLSTM-CNN | 89.21 | 90.48 | 89.84 |
| BLSTM-CNN-CRF | 91.01 | 91.19 | 91.10 |

注：

 - CoNLL 2003语料下载地址: [CoNLL 2003 NER](https://www.clips.uantwerpen.be/conll2003/ner/)，标注方式需要转换为`BIESO`。
 - 词向量下载地址: [glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)，词向量需修改为word2vec词向量格式，即`txt`文件的首部需要有`'词表大小 向量维度'`信息。

## 4. Requirements

 - python3
    - gensim
    - h5py
    - numpy
    - torch==0.4.0
    - pyyaml


## 5. 参考

1. Lample G, Ballesteros M, et al. [Neural Architectures for Named Entity Recognition](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf). NANCL, 2016.

2. Ma X, and Hovy E. [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf). ACL, 2016.

## Updating...

 - `clip`: RNN层的梯度裁剪；

 - `deterministic`: 模型的可重现性；

 - one-hot编码字符向量；

 - lstm抽取字符层面特征；

 - 单机多卡训练。
