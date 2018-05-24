#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""处理conllu格式的文件
"""
import re
import codecs


def read_conllu(path, zip_format=True):
    """读取conllu格式文件
    Args:
         path: str

    yield:
        list(list)
    """
    pattern_space = re.compile('\s+')
    feature_items = []
    file_data = codecs.open(path, 'r', encoding='utf-8')
    line = file_data.readline()
    line_idx = 1
    while line:
        line = line.strip()
        if not line:
            # 判断是否存在多个空行
            if not feature_items:
                print('存在多个空行！`{0}` line: {1}'.format(path, line_idx))
                exit()

            # 处理上一个实例
            if zip_format:
                yield list(zip(*feature_items))
            else:
                yield feature_items

            line = file_data.readline()
            line_idx += 1
            feature_items = []
        else:
            # 记录特征
            items = pattern_space.split(line)
            feature_items.append(items)

            line = file_data.readline()
            line_idx += 1
    # the last one
    if feature_items:
        if zip_format:
            yield list(zip(*feature_items))
        else:
            yield feature_items
    file_data.close()
