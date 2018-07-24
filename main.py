#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Usage:

1. preprocessing and train

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/demo.train.yml -p --train

2. train

若已经完成了预处理，则可以直接进行模型训练:

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/demo.train.yml --train

3. test

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./configs/demo.train.yml --test

"""
import os
import sys
import codecs
from string import ascii_letters, digits
from collections import Counter
import yaml
import h5py
import numpy as np

from optparse import OptionParser

from sltk.preprocessing import normalize_word

from sltk.utils import read_conllu
from sltk.utils import build_word_embed
from sltk.utils import tokens2id_array
from sltk.utils import check_parent_dir, object2pkl_file, read_bin

from sltk.data import DataIter, DataUtil

from sltk.nn.modules import SLModel
from sltk.train import SLTrainer
from sltk.infer import Inference

import torch
import torch.optim as optim


def parse_opts():
    op = OptionParser()
    op.add_option(
        '-c', '--config', dest='config', type='str', help='配置文件路径')
    op.add_option('--train', dest='train', action='store_true', default=True, help='训练模式')
    op.add_option('--test', dest='test', action='store_true', default=False, help='测试模式')
    op.add_option(
        '-p', '--preprocess', dest='preprocess', action='store_true', default=False, help='是否进行预处理')
    argv = [] if not hasattr(sys.modules['__main__'], '__file__') else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if not opts.config:
        op.print_help()
        exit()
    if opts.test:
        opts.train = False
    return opts


def update_feature_dict(tokens_list, feature_dict, feature_cols, feature_names,
                        normalize=True, has_label=True):
    """
    更新特征字典
    Args:
        tokens_list: list(list)
        feature_dict: dict
        feature_cols: list(int)
        feature_names: list(str)
        normalize: bool, 是否标准化单词
        has_label: bool
    """
    for i, col in enumerate(feature_cols):
        for token in tokens_list[col]:
            if normalize:
                token = normalize_word(token)
            feature_dict[feature_names[i]].update([token])
    if has_label:
        for label in tokens_list[-1]:
            feature_dict['label'].add(label)


def extract_feature_dict(path_data, feature_cols, feature_names, feature_dict,
                         sentence_lens=None, normalize=True, has_label=True):
    """从数据中统计特征
    Args:
        path_data: str, 数据路径
        feature_cols: list(int), 特征的列数
        feature_names: list(str), 特征名称
        feature_dict: dict
        sentence_lens: list, 用于记录句子长度
        normalize: bool, 是否标准化单词
        has_label: bool, 数据是否带有标签
    """
    data_idx = 0
    for i, tokens_list in enumerate(read_conllu(path_data)):
        sys.stdout.write('`{0}`: {1}\r'.format(path_data, i))
        sys.stdout.flush()
        update_feature_dict(
            tokens_list, feature_dict, feature_cols, feature_names,
            normalize=normalize, has_label=has_label)
        sentence_lens.append(len(tokens_list[0]))
        data_idx += 1
    return data_idx


def data2hdf5(path_data, data_count, feature_cols, feature_names, token2id_dict,
              use_char=False, max_word_len=None, has_label=True, normalize=True):
    """将数据转为id形式, 存入hdf5格式文件
    Args:
        path_data: 原始文件路径
        data_count: int, 数据量
        feature_cols: list(int), 特征的列数
        feature_names: list(str), 特征名称
        token2id_dict: dict
        use_char: bool, 是否使用char feature
        max_word_len: int, 单词最大长度, 用作提取char feature
        has_label: bool, 数据是否带有标签
        normalize: bool, 是否标准化单词
    """
    def padding_char(word, max_word_len):
        """
        截图长单词、补全短单词
        Args:
            word: str
            max_word_len: int, 单词最大长度
        Return:
            word: str
        """
        if len(word) > max_word_len:
            half = int(max_word_len // 2)
            word = word[:half] + word[-(max_word_len-half):]
            return word
        return word + ' ' * (max_word_len - len(word))

    # 初始化hdf5文件
    path_hdf5 = path_data + '.hdf5'
    file_hdf5 = h5py.File(path_hdf5, 'w')
    dt = h5py.special_dtype(vlen=np.dtype(np.int32).type)
    dataset_dict = dict()
    for feature_name in feature_names:
        dataset = file_hdf5.create_dataset(feature_name, shape=(data_count,), dtype=dt)
        dataset_dict[feature_name] = dataset
    if use_char:
        dataset_char = file_hdf5.create_dataset('char', shape=(data_count,), dtype=dt)
        dataset_dict['char'] = dataset_char
    dataset_label = file_hdf5.create_dataset('label', shape=(data_count,), dtype=dt)
    dataset_dict['label'] = dataset_label

    for i, tokens_list in enumerate(read_conllu(path_data)):
        sys.stdout.write('`{0}`: {1}\r'.format(path_hdf5, i))
        sys.stdout.flush()
        for j, col in enumerate(feature_cols):
            feature_name = feature_names[j]
            tokens = tokens_list[col]
            if normalize:  # normalize
                tokens = [normalize_word(token) for token in tokens]
            token_arr = tokens2id_array(tokens, token2id_dict[feature_name])
            dataset_dict[feature_name][i] = token_arr
        if use_char:  # 提取char feature
            words = ''.join([padding_char(word, max_word_len) for word in tokens_list[0]])
            char_arr = tokens2id_array(words, token2id_dict['char'])
            dataset_dict['char'][i] = char_arr
        if has_label:
            label_arr = tokens2id_array(tokens_list[-1], token2id_dict['label'])
            dataset_dict['label'][i] = label_arr
    sys.stdout.write('`{0}`: {1}\n'.format(path_hdf5, i+1))
    sys.stdout.flush()

    file_hdf5.close()


def preprocessing(configs):
    """预处理
    Args:
        configs: yaml configuration object
    """
    path_train = configs['data_params']['path_train']
    path_dev = configs['data_params']['path_dev'] if 'path_dev' in configs['data_params'] else None
    path_test = configs['data_params']['path_test'] if 'path_test' in configs['data_params'] else None

    feature_cols = configs['data_params']['feature_cols']
    feature_names = configs['data_params']['feature_names']
    min_counts = configs['data_params']['alphabet_params']['min_counts']
    root_alphabet = configs['data_params']['alphabet_params']['path']
    path_pretrain_list = configs['data_params']['path_pretrain']

    use_char = configs['model_params']['use_char']
    max_word_len = configs['model_params']['char_max_len']

    normalize = configs['word_norm']

    feature_dict = {}
    for feature_name in feature_names:
        feature_dict[feature_name] = Counter()
    feature_dict['label'] = set()
    sentence_lens = []

    # 处理训练、开发、测试数据
    print('读取文件...')
    data_count_train = extract_feature_dict(
        path_train, feature_cols, feature_names, feature_dict, sentence_lens,
        normalize=normalize, has_label=True, )
    print('`{0}`: {1}'.format(path_train, data_count_train))
    if path_dev:
        data_count_dev = extract_feature_dict(
            path_dev, feature_cols, feature_names, feature_dict, sentence_lens,
            normalize=normalize, has_label=True)
        print('`{0}`: {1}'.format(path_dev, data_count_dev))
    if path_test:
        data_count_test = extract_feature_dict(
            path_test, feature_cols, feature_names, feature_dict, sentence_lens,
            normalize=normalize, has_label=False)
        print('`{0}`: {1}'.format(path_test, data_count_test))

    # for name in feature_dict:
    #     print(name, len(feature_dict[name]))

    # 构建label alphabet
    token2id_dict = dict()
    label2id_dict = dict()
    for label_idx, label in enumerate(sorted(feature_dict['label'])):
        label2id_dict[label] = label_idx + 1  # 从1开始编号
    token2id_dict['label'] = label2id_dict
    path_label2id_pkl = os.path.join(root_alphabet, 'label.pkl')
    check_parent_dir(path_label2id_pkl)
    object2pkl_file(path_label2id_pkl, label2id_dict)

    # 构建特征alphabet
    for i, feature_name in enumerate(feature_names):
        feature2id_dict = dict()
        start_idx = 1
        for item in sorted(feature_dict[feature_name].items(), key=lambda d: d[1], reverse=True):
            if item[1] < min_counts[i]:
                continue
            feature2id_dict[item[0]] = start_idx
            start_idx += 1
        token2id_dict[feature_name] = feature2id_dict
        # write to file
        object2pkl_file(
            os.path.join(root_alphabet, '{0}.pkl'.format(feature_name)), feature2id_dict)

    # 构建char alphabet
    if use_char:
        char2id_dict = {}
        for i, c in enumerate(ascii_letters + digits):
            char2id_dict[c] = i + 2
        char2id_dict[' '] = 0
        token2id_dict['char'] = char2id_dict
        object2pkl_file(os.path.join(root_alphabet, 'char.pkl'), char2id_dict)

    # 构建embedding table
    print('抽取预训练词向量...')
    for i, feature_name in enumerate(feature_names):
        if path_pretrain_list[i]:
            print('特征`{0}`使用预训练词向量`{1}`:'.format(feature_name, path_pretrain_list[i]))
            word_embed_table, exact_match_count, fuzzy_match_count, unknown_count, \
                total_count = build_word_embed(token2id_dict[feature_name], path_pretrain_list[i])
            print('\t精确匹配: {0} / {1}'.format(exact_match_count, total_count))
            print('\t模糊匹配: {0} / {1}'.format(fuzzy_match_count, total_count))
            print('\tOOV: {0} / {1}'.format(unknown_count, total_count))
            # write to file
            path_pkl = os.path.join(os.path.dirname(path_pretrain_list[i]), '{0}.embed.pkl'.format(feature_name))
            object2pkl_file(path_pkl, word_embed_table)

    # 将数据转为id形式，存入hdf5文件
    print('convert data to hdf5...')
    data2hdf5(
        path_train, data_count_train, feature_cols, feature_names,
        token2id_dict, use_char, max_word_len, has_label=True, normalize=normalize)
    if path_dev:
        data2hdf5(path_dev, data_count_dev, feature_cols, feature_names,
                  token2id_dict, use_char, max_word_len, has_label=True, normalize=normalize)
    if path_test:
        data2hdf5(path_test, data_count_test, feature_cols, feature_names,
                  token2id_dict, use_char, max_word_len, has_label=False, normalize=normalize)


def init_model(configs):
    """初始化模型
    Returns:
        model: SLModel
    """
    use_char = configs['model_params']['use_char']

    feature_names = configs['data_params']['feature_names']
    # init feature alphabet size dict
    feature_size_dict = dict()
    root_alphabet = configs['data_params']['alphabet_params']['path']
    for feature_name in feature_names:
        alphabet = read_bin(os.path.join(root_alphabet, '{0}.pkl'.format(feature_name)))
        feature_size_dict[feature_name] = len(alphabet) + 1
    alphabet = read_bin(os.path.join(root_alphabet, 'label.pkl'))
    feature_size_dict['label'] = len(alphabet) + 1
    if use_char:
        alphabet = read_bin(os.path.join(root_alphabet, 'char.pkl'))
        feature_size_dict['char'] = len(alphabet) + 1

    # init feature dim size dict and pretrain embed dict
    path_pretrain_list = configs['data_params']['path_pretrain']
    embed_sizes = configs['model_params']['embed_sizes']
    feature_dim_dict = dict()
    for i, feature_name in enumerate(feature_names):
        feature_dim_dict[feature_name] = embed_sizes[i]
    pretrained_embed_dict = dict()
    for i, feature_name in enumerate(feature_names):
        if path_pretrain_list[i]:
            path_pkl = os.path.join(os.path.dirname(path_pretrain_list[i]), '{0}.embed.pkl'.format(feature_name))
            embed = read_bin(path_pkl)
            feature_dim_dict[feature_name] = embed.shape[-1]
            pretrained_embed_dict[feature_name] = embed
    if use_char:
        feature_dim_dict['char'] = configs['model_params']['char_dim']

    # init requires_grad_dict
    require_grads = configs['model_params']['require_grads']
    require_grad_dict = {}
    for i, feature_name in enumerate(feature_names):
        require_grad_dict[feature_name] = require_grads[i]
    if use_char:
        require_grad_dict['char'] = configs['model_params']['char_requires_grad']

    # init char parameters
    filter_sizes = configs['model_params']['conv_filter_sizes']
    filter_nums = configs['model_params']['conv_filter_nums']

    # init rnn parameters
    rnn_unit_type = configs['model_params']['rnn_type']
    num_rnn_units = configs['model_params']['rnn_units']
    num_layers = configs['model_params']['rnn_layers']
    bi_flag = configs['model_params']['bi_flag']

    use_crf = configs['model_params']['use_crf']

    # init other parameters
    dropout_rate = configs['model_params']['dropout_rate']
    average_batch = configs['model_params']['average_batch']
    deterministic = configs['model_params']['deterministic']
    use_cuda = configs['model_params']['use_cuda']

    # init model
    sl_model = SLModel(
        feature_names=feature_names, feature_size_dict=feature_size_dict, feature_dim_dict=feature_dim_dict,
        pretrained_embed_dict=pretrained_embed_dict, require_grad_dict=require_grad_dict, use_char=use_char,
        filter_sizes=filter_sizes, filter_nums=filter_nums, rnn_unit_type=rnn_unit_type, num_rnn_units=num_rnn_units,
        num_layers=num_layers, bi_flag=bi_flag, dropout_rate=dropout_rate, average_batch=average_batch,
        use_crf=use_crf, use_cuda=use_cuda)

    if deterministic:  # for deterministic
        torch.backends.cudnn.enabled = False

    use_cuda = configs['model_params']['use_cuda']
    if use_cuda:
        sl_model = sl_model.cuda()

    return sl_model


def init_train_data(configs):
    """初始化训练数据
    Returns:
        data_iter_train: DataIter
        data_iter_dev: DataIter
    """
    all_in_memory = configs['all_in_memory']
    char_max_len = configs['model_params']['char_max_len']
    batch_size = configs['model_params']['batch_size']
    dev_size = configs['model_params']['dev_size']
    max_len_limit = configs['max_len_limit']

    features_names = configs['data_params']['feature_names']
    data_names = [name for name in features_names]
    use_char = configs['model_params']['use_char']
    if use_char:
        data_names.append('char')
    data_names.append('label')

    # load train hdf5 file
    path_data = configs['data_params']['path_train'] + '.hdf5'
    train_object_dict_ = h5py.File(path_data, 'r')
    train_object_dict = train_object_dict_
    if all_in_memory:
        train_object_dict = dict()
        for data_name in data_names:  # 全部加载到内存
            train_object_dict[data_name] = train_object_dict_[data_name].value
    train_count = train_object_dict[data_names[0]].size

    # load dev hdf5 file
    if 'path_dev' not in configs['data_params'] or not configs['data_params']['path_dev']:
        # 拆分训练集
        data_utils = DataUtil(
            train_count, train_object_dict, data_names, use_char=use_char, char_max_len=char_max_len,
            batch_size=batch_size, max_len_limit=max_len_limit)
        data_iter_train, data_iter_dev = data_utils.split_dataset(proportions=(1-dev_size, dev_size), shuffle=False)
    else:
        path_data = configs['data_params']['path_dev'] + '.hdf5'
        dev_object_dict_ = h5py.File(path_data, 'r')
        dev_object_dict = train_object_dict_
        if all_in_memory:
            dev_object_dict = dict()
            for data_name in data_names:  # 全部加载到内存
                dev_object_dict[data_name] = dev_object_dict_[data_name].value
        dev_count = dev_object_dict[data_names[0]].size
        data_iter_dev = DataIter(
            dev_count, dev_object_dict, data_names, use_char=use_char, char_max_len=char_max_len,
            batch_size=batch_size, max_len_limit=max_len_limit)
        data_iter_train = DataIter(
            train_count, train_object_dict, data_names, use_char=use_char, char_max_len=char_max_len,
            batch_size=batch_size, max_len_limit=max_len_limit)

    return data_iter_train, data_iter_dev


def init_test_data(configs):
    """初始化训练数据
    Returns:
        data_iter_train: DataIter
        data_iter_dev: DataIter
    """
    all_in_memory = configs['all_in_memory']
    char_max_len = configs['model_params']['char_max_len']
    batch_size = configs['model_params']['batch_size']
    dev_size = configs['model_params']['dev_size']
    max_len_limit = configs['max_len_limit']

    features_names = configs['data_params']['feature_names']
    data_names = [name for name in features_names]
    use_char = configs['model_params']['use_char']
    if use_char:
        data_names.append('char')
    data_names.append('label')

    # load train hdf5 file
    path_data = configs['data_params']['path_test'] + '.hdf5'
    test_object_dict_ = h5py.File(path_data, 'r')
    test_object_dict = test_object_dict_
    if all_in_memory:
        test_object_dict = dict()
        for data_name in data_names:  # 全部加载到内存
            test_object_dict[data_name] = test_object_dict_[data_name].value
    test_count = test_object_dict[data_names[0]].size

    data_iter = DataIter(
        test_count, test_object_dict, data_names, use_char=use_char, char_max_len=char_max_len,
        batch_size=batch_size, max_len_limit=max_len_limit)

    return data_iter


def init_optimizer(configs, model):
    """初始化optimizer
    Returns:
        optimizer
    """
    optimizer_type = configs['model_params']['optimizer']
    learning_rate = configs['model_params']['learning_rate']
    l2_rate = configs['model_params']['l2_rate']
    momentum = configs['model_params']['momentum']
    lr_decay = 0
    # 过滤不需要更新参数的
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type.lower() == "sgd":
        lr_decay = configs['model_params']['lr_decay']
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=l2_rate)
    elif optimizer_type.lower() == "adagrad":
        optimizer = optim.Adagrad(parameters, lr=learning_rate, weight_decay=l2_rate)
    elif optimizer_type.lower() == "adadelta":
        optimizer = optim.Adadelta(parameters, lr=learning_rate, weight_decay=l2_rate)
    elif optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=l2_rate)
    elif optimizer_type.lower() == "adam":
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=l2_rate)
    else:
        print('请选择正确的optimizer: {0}'.format(optimizer_type))
        exit()
    return optimizer, lr_decay


def init_trainer(configs, data_iter_train, data_iter_dev, model, optimizer, lr_decay):
    """初始化model trainer
    Returns:
        trainer: SLTrainer
    """
    feature_names = configs['data_params']['feature_names']
    use_char = configs['model_params']['use_char']
    max_len_char = configs['model_params']['char_max_len']
    path_save_model = configs['data_params']['path_model']
    check_parent_dir(path_save_model)

    nb_epoch = configs['model_params']['nb_epoch']
    max_patience = configs['model_params']['max_patience']

    learning_rate = configs['model_params']['learning_rate']

    trainer = SLTrainer(
        data_iter_train=data_iter_train, data_iter_dev=data_iter_dev, feature_names=feature_names,
        use_char=use_char, max_len_char=max_len_char, model=model, optimizer=optimizer,
        path_save_model=path_save_model, nb_epoch=nb_epoch, max_patience=max_patience,
        learning_rate=learning_rate, lr_decay=lr_decay)

    return trainer


def load_model(configs):
    """加载预训练的model
    """
    model = init_model(configs)

    path_model = configs['data_params']['path_model']
    model_state = torch.load(path_model)
    model.load_state_dict(model_state)
    return model


def train_model(configs):
    """训练模型
    """
    # init model
    sl_model = init_model(configs)
    print(sl_model)

    # init data
    data_iter_train, data_iter_dev = init_train_data(configs)

    # init optimizer
    optimizer, lr_decay = init_optimizer(configs, model=sl_model)

    # init trainer
    model_trainer = init_trainer(
        configs, data_iter_train, data_iter_dev, sl_model, optimizer, lr_decay)

    model_trainer.fit()


def test_model(configs):
    """测试模型
    """
    # init model
    model = load_model(configs)

    # init test data
    data_iter_test = init_test_data(configs)

    # init infer
    path_conllu_test = configs['data_params']['path_test']
    if 'path_test_result' not in configs['data_params'] or \
       not configs['data_params']['path_test_result']:
        path_result = configs['data_params']['path_test'] + '.result'
    else:
        path_result = configs['data_params']['path_test_result']
    # label to id dict
    path_pkl = os.path.join(configs['data_params']['alphabet_params']['path'], 'label.pkl')
    label2id_dict = read_bin(path_pkl)
    infer = Inference(
        model=model, data_iter=data_iter_test, path_conllu=path_conllu_test,
        path_result=path_result, label2id_dict=label2id_dict)

    # do infer
    infer.infer2file()


def main():
    opts = parse_opts()
    configs = yaml.load(codecs.open(opts.config, encoding='utf-8'))

    if opts.train:  # train
        # 判断是否需要预处理
        if opts.preprocess:
            preprocessing(configs)
        # 训练
        train_model(configs)
    else:  # test
        test_model(configs)


if __name__ == '__main__':
    main()
