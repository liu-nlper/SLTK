#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
import random
import numpy as np


class DataUtil(object):

    def __init__(self, data_count, data_object_dict, data_names, use_char=False, char_max_len=None,
                 data_type_dict=None, batch_size=32, max_len_limit=None, seed=1337):
        """数据集划分工具类.
        Args:
            data_count: int, 数据总数;
            data_object_dict: dict({str, data_object}), 数据名: 数据, 数据格式为np.array或可根据下标
                索引的数据类型(如hdf5格式), 数据类型为int或np.int32, 单个实例的不同特征长度应相同;
            data_names: list(str), 数据名称, e.g., ['f1', 'f2', 'label'];
            use_char: bool, 是否使用char feature
            char_max_len: int, 单词最大长度
            data_type_dict: dict({str: type}), e.g., {'f1': np.int32, 'label': np.int32},
                default is np.int32
            batch_size: int, batch size, default is 32;
            max_len_limit: int, batch的最大长度限制(该值非max length),
                若是None(default), 则按照batch中最长的实例为准;
            seed: int, random seed, default is 1337.

        Notes:
            data_object_dict中单个实例的所有特征长度必须相等.
        """
        self._data_count = data_count
        self._data_ids = list(range(self._data_count))
        self._data_object_dict = data_object_dict
        self._data_names = data_names
        self._use_char = use_char
        self._char_max_len = char_max_len
        self._batch_size = batch_size
        self._max_len_limit = max_len_limit
        self._seed = seed

        # init data type
        self._data_type_dict = data_type_dict
        if not self._data_type_dict:
            self._data_type_dict = dict()
            for data_name in self._data_names:
                self._data_type_dict[data_name] = np.int32
        for data_name in self._data_names:
            if data_name not in self._data_type_dict:
                self._data_type_dict[data_name] = np.int32

    def split_dataset(self, proportions=(4, 1), shuffle=False):
        """分化数据集.
        Args:
            proportions: tuple(int), 划分数据集的比例, 例如(4, 1)表示将数据集划分为80%和20%;
                (7, 2, 1)表示划分为70%, 20%和10%;
            shuffle: bool, 是否打乱数据

        Returns:
            data_iter_list: DataIter object list.
        """
        if shuffle:
            random.seed(self._seed)
            random.shuffle(self._data_ids)
        proportions_ = np.array(proportions) / float(sum(proportions))
        data_sizes = (proportions_ * self._data_count).astype(np.int32)
        data_iter_list = []
        current_count = 0
        for i in range(len(proportions)):
            start, end = current_count, current_count + data_sizes[i]
            # 构建data iter
            data_iter = DataIter(
                data_sizes[i], self._data_object_dict, self._data_names, self._use_char, self._char_max_len,
                self._data_type_dict, self._batch_size, self._max_len_limit, self._seed)
            data_iter.data_ids = self._data_ids[start: end]  # reset data_ids
            data_iter_list.append(data_iter)

        return data_iter_list


class DataIter(object):

    def __init__(self, data_count, data_object_dict, data_names, use_char=False, char_max_len=None,
                 data_type_dict=None, batch_size=32, max_len_limit=None, seed=1337):
        """数据迭代器.
        Args:
            data_count: int, 数据总数;
            data_object_dict: dict({str, data_object}), 数据名: 数据, 数据格式为np.array或可根据下标
                索引的数据类型(如hdf5格式), 数据类型为int或np.int32, 单个实例的不同特征长度应相同;
            data_names: list(str), 数据名称, e.g., ['f1', 'f2', 'label'];
            use_char: bool, 是否使用char feature
            char_max_len: int, 单词最大长度
            data_type_dict: dict({str: type}), e.g., {'f1': np.int32, 'label': np.int32},
                default is np.int32
            batch_size: int, batch size, default is 32;
            max_len_limit: int, batch的最大长度限制(该值非max length),
                若是None(default), 则按照batch中最长的实例为准;
            seed: int, random seed, default is 1337.

        Notes:
            data_object_dict中单个实例的所有特征长度必须相等.
        """
        self._data_count = data_count
        self._data_ids = list(range(self._data_count))
        self._data_object_dict = data_object_dict
        self._data_names = data_names
        self._use_char = use_char
        self._char_max_len = char_max_len
        self._batch_size = batch_size
        self._max_len_limit = max_len_limit
        self._seed = seed

        self._iter_count = math.ceil(self._data_count/float(self._batch_size))

        # init data type
        self._data_type_dict = data_type_dict
        if not self._data_type_dict:
            self._data_type_dict = dict()
            for data_name in self._data_names:
                self._data_type_dict[data_name] = np.int32
        for data_name in self._data_names:
            if data_name not in self._data_type_dict:
                self._data_type_dict[data_name] = np.int32

        # iter variable
        self._iter_variable = 0

    def shuffle(self):
        """shuffle data."""
        random.seed(self._seed)
        random.shuffle(self._data_ids)

    def _generate_batch(self, start, end):
        """产生批量的数据.
        Args:
            start: int, 数据起始位置
            end: int, 数据结束位置

        Returns:
            pass
        """
        batch_size = end - start

        # 计算批量的最大长度
        batch_max_len = max([len(item) for item in self._data_object_dict[self._data_names[0]][self._data_ids[start:end]]])
        if self._max_len_limit:
            batch_max_len = batch_max_len if batch_max_len <= self._max_len_limit else self._max_len_limit
            if self._use_char:
                batch_char_max_len = batch_max_len * self._char_max_len

        # 生成数据
        batch_dict = dict()
        for data_name in self._data_names:
            dtype = self._data_type_dict[data_name]
            batch_dict[data_name] = np.zeros((batch_size, batch_max_len), dtype=dtype)

            data_object = self._data_object_dict[data_name]
            for i, item in enumerate(data_object[self._data_ids[start:end]]):
                len_item = len(item)
                len_item = len_item if len_item <= batch_max_len else batch_max_len
                batch_dict[data_name][i][:len_item] = item[:len_item]
        # char feature
        if self._use_char:
            batch_dict['char'] = np.zeros((batch_size, batch_max_len*self._char_max_len), dtype=np.int32)
            data_object = self._data_object_dict['char']
            for i, item in enumerate(data_object[self._data_ids[start:end]]):
                # print(item.shape)
                len_item = len(item)
                len_item = len_item if len_item <= batch_char_max_len else batch_char_max_len
                batch_dict['char'][i][:len_item] = item[:len_item]

        return batch_dict

    @property
    def char_max_len(self):
        return self._char_max_len

    @property
    def iter_count(self):
        return self._iter_count

    @property
    def data_count(self):
        return self._data_count

    @data_count.setter
    def data_count(self, value):
        self._data_count = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def data_ids(self):
        return self._data_ids

    @data_ids.setter
    def data_ids(self, value):
        self._data_ids = value

    @property
    def iter_variable(self):
        return self._iter_variable

    def __len__(self):
        return self._data_count

    def __iter__(self):
        self._iter_variable = 0
        return self

    def __next__(self):
        start = self._iter_variable
        end = self._iter_variable + self._batch_size
        if end > self._data_count:
            end = self._data_count
        if self._iter_variable > self._data_count or start >= end:
            self.shuffle()
            raise StopIteration()
        self._iter_variable = end
        return self._generate_batch(start, end)
