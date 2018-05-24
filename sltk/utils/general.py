#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np


def tokens2id_array(items, voc, oov_id=1):
    """
    将词序列映射为id序列
    Args:
        items: list, 词序列
        voc: item -> id的映射表
        oov_id: int, 未登录词的编号, default is 1
    Returns:
        arr: np.array, shape=[max_len,]
    """
    arr = np.zeros((len(items),), dtype='int32')
    for i in range(len(items)):
        if items[i] in voc:
            arr[i] = voc[items[i]]
        else:
            arr[i] = oov_id
    return arr
