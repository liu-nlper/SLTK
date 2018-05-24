#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""测试加载数据集
"""
import sys
import h5py


root_project = '/home/ljx/Workspace/PythonProjects/SLTK_v1.0'
sys.path.append(root_project)

from sltk.data import DataIter


data_names = ['word', 'pos', 'chunk', 'char', 'label']

# load hdf5 file
data_object_dict_ = h5py.File('../data/train.txt.hdf5', 'r')
data_object_dict = dict()
for data_name in data_names:  # 全部加载到内存
    data_object_dict[data_name] = data_object_dict_[data_name].value

data_count = data_object_dict[data_names[0]].size

data_iter = DataIter(
    data_count, data_object_dict, data_names, use_char=True, char_max_len=15,
    batch_size=32, max_len_limit=100, seed=1337)
for data in data_iter:
    print(data['word'].shape)
