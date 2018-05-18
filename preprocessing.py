#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    预处理:
        1. 生成特征字典
        2. 将训练数据切分成小文件
"""
import os
import re
import sys
import codecs
import pickle
import numpy as np
from optparse import OptionParser
from collections import Counter
from TorchNN.utils import build_word_voc, build_word_embed
from TorchNN.utils import is_interactive, parse_int_list


def update_feature_dict(items, feature_dict, features, has_label=True):
    """
    更新特征字典

    Args:
        items: list of str
        feature_dict: dict
        features: list of int
    """
    for feature_i in features:
        feature_dict[feature_i].update([items[feature_i]])
    if has_label:
        feature_dict['label'].add(items[-1])


def list2file(feature_items, file_name):
    """
    将feature_items写入文件

    Args:
        feature_items: list of str
        file_name: str, 文件名
    """
    file_w = codecs.open(file_name, 'w', encoding='utf-8')
    for item in feature_items:
        file_w.write('{0}\n'.format(item))
    file_w.close()


def precessing_data(path_data, has_label, features, root_data_idx, feature_dict=None,
                    sentence_lens=None):
    """
    Args:
        path_data: str, 数据路径
        has_label: bool, 数据是否带有标签
        features: list of int, 特征所在的列, default is [0]
        root_data_idx: str, 数据索引根目录
        feature_dict: dict, 统计特征、label
        sentence_lens: list, 记录句子长度

    Returns:
        data_idx: int, 处理的实例数量
    """
    pattern_feature = re.compile('\s')

    data_idx = 0  # 数据编号
    feature_items = []

    if not os.path.exists(root_data_idx):
        os.mkdir(root_data_idx)
    file_data = codecs.open(path_data, 'r', encoding='utf-8')
    line = file_data.readline()
    line_idx = 1
    while line:
        line = line.strip()
        if not line:
            # 判断是否存在多个空行
            if not feature_items:
                print('存在多个空行！`{0}` line: {1}'.format(path_data, line_idx))
                exit()

            # 处理上一个实例
            file_name = os.path.join(root_data_idx, '{0}.txt'.format(data_idx))
            list2file(feature_items, file_name)
            sentence_lens.append(len(feature_items))
            data_idx += 1
            sys.stdout.write('处理实例数: {0}\r'.format(data_idx))
            sys.stdout.flush()

            line = file_data.readline()
            line_idx += 1
            feature_items = []
        else:
            feature_items.append(line)
            # 记录特征
            items = pattern_feature.split(line)
            update_feature_dict(items, feature_dict, features, has_label=has_label)

            line = file_data.readline()
            line_idx += 1

    # the last one
    if feature_items:
        file_name = os.path.join(root_data_idx, '{0}.txt'.format(data_idx))
        list2file(feature_items, file_name)
        sentence_lens.append(len(feature_items))
        data_idx += 1

    file_data.close()

    path_idx_num = os.path.join(root_data_idx, 'nums.txt')
    file_idx_num = codecs.open(path_idx_num, 'w', encoding='utf-8')
    # 写入文件
    file_idx_num.write('{0}'.format(data_idx))

    return data_idx


def preprocessing(path_data_train, path_data_dev=None, path_data_test=None, features=[0],
                  root_train_idx=None, root_dev_idx=None, root_test_idx=None, root_voc=None,
                  root_embed=None, path_embed=None, percentile=98):
    """
    Args:
        path_data_train: str, 训练数据路径
        path_data_dev: str, 测试数据路径
        path_data_test: str, 开发数据路径
        features: list of int, 特征所在的列, default is [0]
        root_train_idx: str, 训练数据索引根目录
        root_dev_idx: str, 开发数据索引根目录
        root_test_idx: str, 开发数据索引根目录
        root_voc: str, 词表根目录
        root_embed: str, embedding表根目录
        path_embed: str, embed文件的路径(txt or bin格式)
        percentile: int, 百分位, default is 98
    """
    feature_dict = {}
    for feature_i in features:
        feature_dict[feature_i] = Counter()
    feature_dict['label'] = set()

    sentence_lens = []  # 记录句子长度

    # 处理训练数据
    data_count_train = precessing_data(
        path_data_train, has_label=True, features=features, root_data_idx=root_train_idx,
        feature_dict=feature_dict, sentence_lens=sentence_lens)
    print('训练集实例数: {0}'.format(data_count_train))

    # 处理开发数据、测试数据
    if path_data_dev:
        data_count_dev = precessing_data(
            path_data_dev, has_label=True, features=features, root_data_idx=root_dev_idx,
            feature_dict=feature_dict, sentence_lens=sentence_lens)
        print('开发集实例数: {0}'.format(data_count_dev))

    if path_data_test:
        data_count_test = precessing_data(
            path_data_test, has_label=False, features=features, root_data_idx=root_test_idx,
            feature_dict=feature_dict, sentence_lens=sentence_lens)
        print('测试集实例数: {0}'.format(data_count_test))

    # 构建vocs
    if not os.path.exists(root_voc):
        os.makedirs(root_voc)
    # 构建feature voc, 默认从1开始编号
    for i, feature_i in enumerate(features):
        print('构建feature_{0} voc...'.format(feature_i))
        pt = 100 if feature_i != 0 else percentile
        feature2id_dict = build_word_voc(feature_dict[feature_i], percentile=pt)
        if i == 0:
            feature2id_dict[' '] = 0  # 注：空格id为0，为了char feature
            if path_embed:
                first_feature2id_dict = feature2id_dict
        path_feature_voc = os.path.join(root_voc, 'feature_{0}_2id.pkl'.format(feature_i))
        file_feature_voc = codecs.open(path_feature_voc, 'wb')
        pickle.dump(feature2id_dict, file_feature_voc)
        file_feature_voc.close()

    # 构建label voc, 从1开始编号, 0表示padding值
    print('构建label voc...')
    label2id_dict = dict()
    for label_idx, label in enumerate(sorted(feature_dict['label'])):
        label2id_dict[label] = label_idx + 1
    path_label_voc = os.path.join(root_voc, 'label2id.pkl')
    file_label_voc = codecs.open(path_label_voc, 'wb')
    pickle.dump(label2id_dict, file_label_voc)
    file_label_voc.close()

    # 从预训练的文件中构建embedding表
    if path_embed:
        if not os.path.exists(root_embed):
            os.makedirs(root_embed)
        print('构建word embedding表...')
        word_embed_table, unknow_count = build_word_embed(first_feature2id_dict, path_embed)
        print('\t未登录词数量: {0}/{1}'.format(unknow_count, len(first_feature2id_dict)))
        path_word_embed = os.path.join(root_embed, 'word2vec.pkl')
        file_word_embed = codecs.open(path_word_embed, 'wb')
        pickle.dump(word_embed_table, file_word_embed)
        file_word_embed.close()

    # 句子长度分布
    print('\n句子长度分布:')
    option_len_pt = [90, 95, 98, 100]
    for per in option_len_pt:
        tmp = int(np.percentile(sentence_lens, per))
        print('\t{0} percentile: {1}'.format(per, tmp))

    print('\n类别数: {0}\n'.format(len(feature_dict['label'])))

    print('done!')


op = OptionParser()
op.add_option('-l', '--has_label', dest='has_label', action='store_true',
              default=False, help='数据是否带有标签')
op.add_option('-f', dest='features', default=[0], type='str',
              action='callback', callback=parse_int_list, help='使用的特征列数')
op.add_option(
    '--path_data', dest='path_data', type='str', help='训练语料路径(或不带标签的待预测数据)')
op.add_option('--path_dev', dest='path_dev', default=None, type='str', help='开发语料路径')
op.add_option('--path_test', dest='path_test', default=None, type='str', help='测试语料路径')

op.add_option('--root_idx_data', dest='root_idx_data', default='./data/train_idx',
              type='str', help='训练数据索引根目录')
op.add_option('--root_idx_dev', dest='root_idx_dev', default='./data/dev_idx', type='str',
              help='开发数据索引根目录')
op.add_option('--root_idx_test', dest='root_idx_test', default='./data/test_idx', type='str',
              help='测试数据索引根目录')

op.add_option('--rv', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--re', dest='root_embed', default='./res/embed', type='str', help='embed根目录')
op.add_option('--pe', dest='path_embed', default=None, type='str', help='embed文件路径')
op.add_option('--pt', dest='pt', default=98, type='int', help='构建word voc的百分位值')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if not opts.path_data:
    op.print_help()
    exit()


if __name__ == '__main__':
    if opts.has_label:
        preprocessing(
            opts.path_data, opts.path_dev, opts.path_test, opts.features,
            opts.root_idx_data, opts.root_idx_dev, opts.root_idx_test, opts.root_voc,
            opts.root_embed, opts.path_embed, opts.pt)
    else:
        sentence_lens = []
        precessing_data(
            opts.path_data, has_label=False, features=opts.features,
            root_data_idx=opts.root_idx_data, sentence_lens=sentence_lens)

        print('\n句子长度分布:')
        option_len_pt = [90, 95, 98, 100]
        for per in option_len_pt:
            tmp = int(np.percentile(sentence_lens, per))
            print('\t{0} percentile: {1}'.format(per, tmp))
