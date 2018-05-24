#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
用于从预训练词向量构建embedding表
"""


def load_embed_with_gensim(path_embed):
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
        exact_match_count: int, 精确匹配的词数
        fuzzy_match_count: int, 精确匹配的词数
        unknown_count: int, 未匹配的词数
    """
    import numpy as np
    assert path_embed.endswith('bin') or path_embed.endswith('txt')
    word2vec_model, word_dim = load_embed_with_gensim(path_embed)
    word_count = len(word2id_dict) + 1  # 0 is for padding value
    np.random.seed(seed)
    scope = np.sqrt(3. / word_dim)
    word_embed_table = np.random.uniform(
        -scope, scope, size=(word_count, word_dim)).astype('float32')
    exact_match_count, fuzzy_match_count, unknown_count = 0, 0, 0
    for word in word2id_dict:
        if word in word2vec_model.vocab:
            word_embed_table[word2id_dict[word]] = word2vec_model[word]
            exact_match_count += 1
        elif word.lower() in word2vec_model.vocab:
            word_embed_table[word2id_dict[word]] = word2vec_model[word.lower()]
            fuzzy_match_count += 1
        else:
            unknown_count += 1
    total_count = exact_match_count + fuzzy_match_count + unknown_count
    return word_embed_table, exact_match_count, fuzzy_match_count, unknown_count, total_count
