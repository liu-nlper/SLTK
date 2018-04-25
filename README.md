# SLTK - Sequence Labeling Toolkit

序列化标注工具，实现了Bi-LSTM-CRF模型，并利用pytorch实现了高效的数据加载模块，可以完成:

 - **预处理**。包括构建词表、label表，从预训练文件构建word embedding;
 - **训练**。训练序列化标注模型，并保存在开发集上性能最好的一次模型;
 - **测试**。对新的实例进行标注。

## 1. 快速开始

### 1.1 数据格式

训练数据处理成下列形式，特征之间用制表符(或空格)隔开，每行共n列，1至n-1列为特征，最后一列为label，句子之间用**空行**分隔。

    苏   NR   B-ORG
    州   NR   I-ORG
    大   NN   I-ORG
    学   NN   E-ORG
    位   VV   O
    于   VV   O
    江   NR   B-GPE
    苏   NR   I-GPE
    省   NR   E-GPE
    苏   NR   B-GPE
    州   NR   I-GPE
    市   NR   E-GPE

### 1.2 预处理&训练&测试

**Step 1**:

将训练集、开发集、测试集处理成所需格式，放入`./data/`目录下，文件名分别为`train.txt`、`dev_txt`和`test.txt`，其中`dev.txt`是可选的。

**Step 2**:

    $ cd ./script
    $ chmod a+x *.sh
    $ ./preprocessing.sh
    $ ./train.sh
    $ ./test.sh

### 1.3 处理待测试语料

若需要单独处理待测试语料，则修改`./script/preprocessing_raw.py`中的`path_data`和`root_idx_data`参数，然后执行:

    $ ./preprocessing_raw.sh

## 2. 使用说明

### 2.1 预处理

训练文件的预处理包括:

 - 构建词表，即词->id的映射表，以及label表，以`dict`格式存放在`pkl`文件中;
 - 构建embedding表，根据所提供的预训练词向量文件，抽取所需要的向量，对于不存在于预训练文件中的词，则随机初始化。结果以`np.array`的格式存放在`pkl`文件中;
 - 将训练数据按顺序编号，每个实例写入单独的文件中，便于高效加载；
 - 统计句子长度，输出句子长度的[90, 95, 98, 100]百分位值;
 - 输出标签数量。

**运行方式:**

    $ python3 preprocessing.py -l --path_train ./data/train.txt --path_dev ./data/dev.txt --path_test
    ./data/test.txt --root_idx_train ./data/train_idx --root_idx_dev ./data/dev_idx -root_idx_test
    ./data/test_idx --rv ./res/voc/

若需要使用预训练的词向量，则在上述命令之后添加下列命令，其中`path_to_embed_file`是预训练词向量路径，可以是`bin`或`txt`类型的文件:

    --re ./res/embed/ --pe ./path_to_embed_file

**表. 参数说明**

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|l|bool|True|待处理数据是否带有标签|
|f|list|[0]|使用的特征所在的列|
|path_data|str|./data/train.txt|训练集路径(也可指待标注数据)|
|path_data|str|./data/train.txt|训练集路径(也可指待标注数据)|
|path_dev|str|./data/dev.txt|开发集路径|
|path_test|str|./data/test.txt|测试集路径|
|root_idx_train|str|./data/train_idx/|训练数据索引文件根目录|
|root_idx_dev|str|./data/dev_idx/|开发集索引文件根目录|
|root_idx_test|str|./data/test_idx/|测试集索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|pe|str|None|path_embed，预训练的embed文件路径，`bin`或`txt`；若不提供，则随机初始化|
|pt|int|98|percentile，构建词表时的百分位值|

运行`python3 preprocessing.py -h`可打印出帮助信息。

### 2.2 训练

若预处理时`root_idx`等参数使用的是默认值，则在训练时不需要设定相应参数。

**运行方式:**

    $ CUDA_VISIBLE_DEVICES=0 python3 train.py --ml 90 --bs 256 -g

    # 使用第1列和第3列特征
    $ CUDA_VISIBLE_DEVICES=0 python3 train.py -f 0,2 --bs 256 --ml 90 -g

    # 使用第1列和第3列特征，并设置特征维度分别为64和16
    $ CUDA_VISIBLE_DEVICES=0 python3 train.py -f 0,2 --fd 64,16 --bs 256 --ml 90 -g

    # 若是英文，可利用字符特征，加入参数-c、--dc、--fs和--fn参数
    $ CUDA_VISIBLE_DEVICES=0 python3 train.py --bs 256 --ml 90 -c --dc 16 --fs 3,4 --fn 16,16 -g

**参数说明**

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|f|list|[0]|features，训练时所使用的特征所在的列，若多列特征，则用逗号分隔|
|root_idx_train|str|./data/train_idx/|训练集索引文件根目录|
|root_idx_dev|str|./data/dev_idx/|开发集索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|ml|int|50|max_len，句子最大长度|
|ds|float|0.2|dev_size，开发集占比|
|lstm|int|256|lstm unit size，lstm单元数|
|ln|int|2|layer nums，lstm层数|
|fd|list|[64]|feature_dim，各列特征的维度，若多列特征，则用逗号分隔|
|dp|float|0.5|dropout_rate，dropout rate|
|lr|float|0.002|learning_rate，learning rate|
|ne|int|100|nb_epoch，迭代次数|
|mp|int|5|max_patience，最大耐心值，即开发集上性能超过mp次没有提示，则终止训练|
|rm|str|./model/|root_model，模型根目录|
|bs|int|64|batch_size，batch size|
|c|bool|False|是否使用字符特征(针对英文)|
|dim_char|int|16|字符特征维度|
|fs|list|[3]|卷积核尺寸(字符特征)|
|fn|list|[32]|卷积核数量(字符特征)|
|g|bool|False|是否使用GPU加速|
|rg|bool|True|是否更新词向量|
|crf|bool|False|是否使用CRF层|
|device_ids|list|[0]|GPU编号列表|
|nw|int|8|num_worker，加载数据时的线程数|

运行`python3 train.py -h`可打印出帮助信息。

### 2.3 测试

**运行方式:**

    $ CUDA_VISIBLE_DEVICES=0 python3 test.py --bs 256 -g -o ./result.txt

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|ri|str|./data/train_idx/|root_idx，训练数据索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|pm|str|无|path_model，模型路径|
|bs|int|64|batch_size，batch size|
|g|bool|False|是否使用GPU加速|
|nw|int|8|num_worker，加载数据时的线程数|
|device_ids|list|[0]|GPU编号列表|
|o|str|./result.txt|预测结果存放路径|

运行`python3 test.py -h`可打印出帮助信息。

### 2.4 使用字符特征

该特征用于捕获词形层面的特征，在训练时加入`-c`参数。默认根据`--dc`参数，即字符维度构建embedding表；若使用了`--c_binary`参数，则使用one-hot编码作为字符的向量表示，且在模型训练过程中不更新。

### 2.5 使用多GPU训练&测试

**设置说明**

以使用第一块、第二块GPU(编号分别为0,1)为例，运行脚本时，需设置`CUDA_VISIBLE_DEVICES=0,1`，并**同时**设置参数`--device_ids 0,1`。

主要注意的是，目前版本在设置多GPU训练时，第一张显卡编号必须设置为`0`，否则会出错；若只使用单卡训练，则无此限制。

### 2.6 参数搜索

修改`./script/grid_search.py`中待搜索的参数，运行该脚本。

## 3. Requirements

 - gensim==2.3.0
 - numpy==1.13.1
 - torch==0.3.1
 - torchvision==0.1.9
 - tqdm==4.19.4

## 4. 参考

 - [http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html "http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html")
 - [https://github.com/jiesutd/PyTorchSeqLabel](https://github.com/jiesutd/PyTorchSeqLabel "https://github.com/jiesutd/PyTorchSeqLabel")
 - [http://www.aclweb.org/anthology/N16-1030](http://www.aclweb.org/anthology/N16-1030 "http://www.aclweb.org/anthology/N16-1030")

## Updating

 - TODO:
   * 1. 参数搜索;
   * 2. 模型保存方式.
 - 2018-04-16:
   * 加入`c_binary`选项，若使用此选项，则one-hot编码作为字符的向量表示。
 - 2018-03-28:
   * 预测结果追加到测试文件列尾：
   * 检查文件中是否存在多个空行。
 - 2018-03-26:
   * 加入参数搜索功能(全局搜索/随机搜索)，`script/grid_search.py`。
 - 2018-03-23:
   * 修正模型可重现性问题:
       若在GPU上进行conv操作，可能会出现模型结果不可重现，解决方法是`torch.backends.cudnn.enabled=False`
 - 2018-03-15:
   * 加入单机多卡训练功能，详见`README.md`;
   * 训练时加入`crf`参数选项，即是否使用CRF层;
   * 加入单独处理待测试语料功能;
   * 调整`train.py`和`test.py`中的代码风格.
 - 2018-03-14:
   * 修正bugs: 使用nn.ModuleList替换python list和dict；因为list与dict中的权重不会进行更新.
 - 2018-03-06:
   * 修正:词向量的随机初始化时加入随机数种子，确保每次初始化时词向量不变.
 - 2018-03-05:
   * (1) 预处理中加入开发集选项，若没有提供开发集，则训练过程中会按照`ds`参数划分部分训练数据作为开发集；
   * (2) 针对英文数据加入字符层面特征(`-c`选项)，通过`--dc`、`--fs`和`--fn`参数设定相应参数；
   * (3) 设定随机数种子，使模型权重的初始化和数据shuffle方式固定，从而实现相同参数下模型结果不发生变化；
   * (4) 修改`crf.py`中部分代码，是之适配`torch 0.3.1`版本。
 - 2018-02-27:
   * `bilstm.py`和`bilstm_crf.py`移入`TorchNN/layers`模块中，训练时使用Bi-LSTM-CRF模型.
 - 2018-02-26:
   * `utils/`移入`TorchNN/`中；添加`layers`模块，并完成CRF层(未加入模型).
