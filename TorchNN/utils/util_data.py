#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import re
import codecs
import random
from collections import defaultdict
from . import read_csv, items2id_array

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class SentenceDataset(Dataset):

    def __init__(self, nums, root_idx, max_len, features, feature2id_dict,
                 max_len_char=10, use_char_feature=False):
        """
        Args:
            nums: list, 实例编号
            root_idx: str, 索引文件根目录
            max_len: int, 句子最大长度
            features: list of int, 特征列
            feature2id_dict: dict, 特征->id映射字典

            max_len_char: int, 单词的最大长度, default is 10
            ise_char_feature: bool, 是否使用字符特征, default is False
        """
        self.nums = nums
        self.root_idx = root_idx
        self.max_len = max_len
        self.features = features
        self.has_label = 'label' in feature2id_dict
        self.feature2id_dict = feature2id_dict
        self.pattern_feature = re.compile('\s')

        self.max_len_char = max_len_char
        self.use_char_feature = use_char_feature

        # 构建字符->id字典，从2开始编号，编号1为未登录字符, 0为padding值
        # 数字统一转换为数字0
        if self.use_char_feature:
            from string import ascii_letters, digits
            self.char2id_dict = {}
            for i, c in enumerate(ascii_letters + digits):
                self.char2id_dict[c] = i + 2
            self.char2id_dict[' '] = 0

    def shuffle(self, seed=1337):
        random.seed(seed)
        random.shuffle(self.nums)

    def padding_char(self, word, max_len_char):
        """
        截图长单词、补全短单词
        Args:
            word: str
            max_len_char: int, 单词最大长度

        Return:
            word: str
        """
        if len(word) > max_len_char:
            half = int(max_len_char // 2)
            word = word[:half] + word[-(max_len_char-half):]
            return word
        return word + ' ' * (max_len_char - len(word))

    def get_feature_dict(self, idx):
        path_data = os.path.join(self.root_idx, '{0}.txt'.format(self.nums[idx]))
        text = codecs.open(path_data, 'r', encoding='utf-8').read().strip()
        feature_dict_ = defaultdict(list)
        if self.use_char_feature:
            feature_dict_['char_feature'] = ''
        for line in text.split('\n'):
            items = self.pattern_feature.split(line)
            for feature_i in self.features:
                feature_dict_[feature_i].append(items[feature_i])
            if self.has_label:
                feature_dict_['label'].append(items[-1])

            # char feature
            if self.use_char_feature:
                word_pad = self.padding_char(items[0], self.max_len_char)
                feature_dict_['char_feature'] += word_pad

        # 转为np.array
        feature_dict = dict()
        for feature_i in self.features:
            feature_dict[str(feature_i)] = items2id_array(
                feature_dict_[feature_i], self.feature2id_dict[feature_i], self.max_len)
        if self.has_label:
            feature_dict['label'] = items2id_array(
                feature_dict_['label'], self.feature2id_dict['label'], self.max_len)

        # char feature
        if self.use_char_feature:
            char_features = items2id_array(
                feature_dict_['char_feature'], self.char2id_dict, self.max_len_char * self.max_len)
            # shape=[max_len, max_len_char] 
            char_features = char_features.reshape(self.max_len, self.max_len_char)
            feature_dict['char_feature'] = char_features

        return feature_dict

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, idx):
        return self.get_feature_dict(idx)


class SentenceDataUtil():

    def __init__(self, path_num, root_idx, max_len, features, feature2id_dict,
                 shuffle=False, max_len_char=10, use_char_feature=False, seed=1337):
        """
        Args:
            path_num: str, 记录实例数量的文件
            root_idx: str, 索引文件根目录
            max_len: int, 句子最大长度
            features: list of int, 特征列
            feature2id_dict: dict, 特征->id映射字典
            has_label: bool, 数据中是否带标签
            label2id_dict: dict, label->id映射字典
            shuffle: 是否打乱数据集, default is False

            max_len_char: int, 单词的最大长度, default is 10
            ise_char_feature: bool, 是否使用字符特征, default is False

            seed: int, 随机数种子, default is 1337
        """
        instance_count = int(
            codecs.open(path_num, 'r', encoding='utf-8').readline().strip())
        self.nums = list(range(instance_count))
        self.root_idx = root_idx
        self.max_len = max_len
        self.features = features
        self.feature2id_dict = feature2id_dict
        self.has_label = 'label' in feature2id_dict

        self.max_len_char = max_len_char
        self.use_char_feature = use_char_feature

        self.shuffle = shuffle
        self.seed = seed

    def shuffle_data(self):
        random.seed(self.seed)
        random.shuffle(self.nums)

    def split_train_and_dev(self, dev_size=0.2):
        """
        划分训练集和开发集

        Args:
            dev_size: None, or a float value between 0 and 1

        Returns:
            dataset_train: torch.utils.data.Dataset
            dataset_dev: torch.utils.data.Dataset
        """
        if self.shuffle:
            self.shuffle_data()

        boundary = int(len(self.nums) * (1. - dev_size))
        nums_train, nums_dev = self.nums[:boundary], self.nums[boundary:]

        dataset_train = SentenceDataset(
            nums_train, self.root_idx, self.max_len, self.features, self.feature2id_dict,
            self.max_len_char, use_char_feature)

        dataset_dev = SentenceDataset(
            nums_dev, self.root_idx, self.max_len, self.features, self.feature2id_dict,
            self.max_len_char, self.use_char_feature)

        return dataset_train, dataset_dev

    def get_all_data(self):
        """
        获取全部数据集

        Returns:
            dataset: torch.utils.data.Dataset
        """
        if self.shuffle:
            self.shuffle_data()
        if not hasattr(self, 'labels'):
            self.labels = None
        dataset = SentenceDataset(
            self.nums, self.root_idx, self.max_len, self.features, self.feature2id_dict,
            self.max_len_char, self.use_char_feature)
        return dataset
