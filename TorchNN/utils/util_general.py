#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import numpy as np
from functools import cmp_to_key

def sort_word(word_count_1, word_count_2):
    """
    排序单词，首先根据词频排序，词频相等的再根据词本身排序
    Args:
        word_count: [word, word_count]
    """
    if word_count_1[1] < word_count_2[1]:
        return -1
    elif word_count_1[1] > word_count_2[1]:
        return 1
    else:  # 词频相等
        if word_count_1[0] >= word_count_2[0]:
            return -1
        else:
            return 1


def build_word_voc(word_count_dict, percentile=100):
    """
    构建word -> id的映射表

    Args:
        word_count_dict: dict, 健: word, 值: word出现的频次
        percentile: int, 百分位值

    Return:
        word2id_dict: dict, 健: word, 值: word对应的id
    """
    current_count, total_count = 0, sum(word_count_dict.values())
    upper_limit = total_count * percentile / 100.
    word2id_dict = dict()
    word_idx = 1  # 从1开始编号
    for item in sorted(word_count_dict.items(), key=cmp_to_key(sort_word), reverse=True):
        word, count = item[:]
        current_count += count
        if current_count > upper_limit:
            break
        word2id_dict[word] = word_idx
        word_idx += 1
    return word2id_dict


def load_embed_use_gensim(path_embed):
    """
    读取预训练的embedding

    Args:
        path_embed: str, bin or txt

    Returns:
        word_embed_dict: dict, 健: word, 值: np.array, vector
        word_dim: int, 词向量的维度
    """
    from gensim.models.keyedvectors import KeyedVectors
    if path_embed.endswith('bin'):
        word_vectors = KeyedVectors.load_word2vec_format(path_embed, binary=True)
    elif path_embed.endswith('txt'):
        word_vectors = KeyedVectors.load_word2vec_format(path_embed, binary=False)
    else:
        raise ValueError('`path_embed` must be `bin` or `txt` file!')
    return word_vectors, word_vectors.vector_size


def build_word_embed(word2id_dict, path_embed, seed=137):
    """
    从预训练的文件中构建word embedding表

    Args:
        word2id_dict: dict, 健: word, 值: word id
        path_embed: str, 预训练的embedding文件，bin or txt

    Returns:
        word_embed_table: np.array, shape=[word_count, embed_dim]
    """
    import numpy as np
    assert path_embed.endswith('bin') or path_embed.endswith('txt')
    word2vec_model, word_dim = load_embed_use_gensim(path_embed)
    word_count = len(word2id_dict) + 1  # 0 is for padding value
    np.random.seed(seed)
    word_embed_table = np.random.normal(size=(word_count, word_dim)).astype('float32')
    unknow_count = 0
    for word in word2id_dict:
        if word in word2vec_model.vocab:
            word_embed_table[word2id_dict[word]] = word2vec_model[word]
        else:
            unknow_count += 1
    return word_embed_table, unknow_count


def items2id_array(items, voc, max_len, none_item=1):
    """
    将词序列映射为id序列

    Args:
        items: list, 词序列
        voc: item -> id的映射表
        max_len: int, 实例的最大长度
        none_item: int, 未登录词的编号, default is 1

    Returns:
        arr: np.array, shape=[max_len,]
    """
    arr = np.zeros((max_len,), dtype='int64')
    min_range = min(max_len, len(items))
    for i in range(min_range):
        if items[i] in voc:
            arr[i] = voc[items[i]]
        else:
            arr[i] = none_item
    return arr


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


def parse_int_list(option, opt, value, parser):
    int_list = [int(i) for i in value.split(',')]
    setattr(parser.values, option.dest, int_list)
