#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import pickle


def check_parent_dir(path):
    """检查path的父目录是否存在，若不存在，则创建之
    Args:                      
        path: str, file path
    """
    parent_name = os.path.dirname(path)
    if not os.path.exists(parent_name):
        os.makedirs(parent_name)


def object2pkl_file(path_pkl, ob):
    """将python对象写入pkl文件
    Args:
        path_pkl: str, pkl文件路径
        ob: python的list, dict, ...
    """
    with open(path_pkl, 'wb') as file_pkl:
        pickle.dump(ob, file_pkl)


def read_bin(path):
    """读取二进制文件
    Args:
        path: str, 二进制文件路径
    Returns:
        pkl_ob: pkl对象
    """
    file = open(path, 'rb')
    return pickle.load(file)
