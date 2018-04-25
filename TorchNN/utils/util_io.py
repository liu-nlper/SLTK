#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import codecs
import pickle


def read_csv(path, split=','):
    """
    读取csv文件

    Args:
        path: str, csv文件路径
        split: 分隔符号

    Return:
        terms: list
    """
    file_csv = codecs.open(path, 'r', encoding='utf-8')
    line = file_csv.readline()
    terms = []
    while line:
        line = line.strip()
        if not line:
            line = file_csv.readline()
            continue
        terms.append(line.split(split))
        line = file_csv.readline()
    return terms


def read_pkl(path):
    """
    读取pkl文件

    Args:
        path: str, pkl文件路径

    Returns:
        pkl_ob: pkl对象
    """
    file_pkl = codecs.open(path, 'rb')
    return pickle.load(file_pkl)
