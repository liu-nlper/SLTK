#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    grid search
"""
import codecs
import random
from itertools import product
import subprocess


def grid_search(path_grid, sample_rate=1.0, seed=1337):
    """
    grid search

    Args:
        path_grid: str, 开发集性能及模型参数存放路径
        sample_rate: float between (0, 1]
        seed: int, seed for sample
    """
    # 待搜索参数
    lstm_units = [100, 125]#, 150, 175, 200]  # lstm单元数
    learn_rate = [0.01, 0.02]#, 0.03]  # 学习率
    # 待搜索参数 end
    arguments = [lstm_units, learn_rate]
    argument_list = list(product(*arguments))

    # 随机搜索
    assert 0. < sample_rate <= 1.
    argument_count = len(argument_list)
    if sample_rate < 1.:
        argument_range = range(argument_count)
        sample_count = int(sample_rate * argument_count)
        random.seed(seed)
        argument_idx = sorted(random.sample(argument_range, sample_count))
        argument_list = [argument_list[idx] for idx in argument_idx]
        argument_count = sample_count
    print('搜索空间大小: {0}'.format(argument_count))

    for argument in argument_list:
        lstm_unit, learn_rate = argument[:]
        command = '''CUDA_VISIBLE_DEVICES=2 python3 ../train.py \
            -f 0,1 \
            --root_idx_train ../data/train_idx \
            --root_idx_dev ../data/dev_idx \
            --rv ../res/voc \
            --ml 55 \
            --fd 200,32 \
            --re ../res/embed \
            --ds 0.1 \
            --lstm {0} \
            --lr {1} \
            --ne 3 \
            --mp 5 \
            --bs 256 \
            --dp 0.5 \
            --rm ../model \
            -c \
            --dc 16 \
            --fs 3 \
            --fn 32 \
            --device_ids 2 \
            --rg False \
            --crf \
            --path_grid {2} \
            -g
        '''.format(lstm_unit, learn_rate, path_grid)
        code = subprocess.call(command, shell=True)
    print('搜索完成!')


def get_best_arg(path_grid):
    """
        获取最佳参数
    """
    best_dev_loss = 10000.
    best_argument = None
    file_grid_search = codecs.open(path_grid, 'r', encoding='utf-8')
    line = file_grid_search.readline()
    while line:
        line = line.strip()
        if not line:
            break
        arguments_tune = eval(line)
        if arguments_tune['dev_loss'] < best_dev_loss:
            best_argument = arguments_tune
            best_dev_loss = arguments_tune['dev_loss']
        line = file_grid_search.readline()
    print('开发集最低loss: {0}'.format(best_dev_loss))
    print('最佳参数: {0}'.format(best_argument))
    print('调整参数:')
    print('lstm: {0}'.format(best_argument['lstm_units']))
    print('learn_rate: {0}'.format(best_argument['learn_rate']))


if __name__ == '__main__':
    # 存放参数的路径
    path_grid = './grid_search.txt'

    # 采样率
    sample_rate = 1.

    grid_search(path_grid, sample_rate=sample_rate)

    get_best_arg(path_grid)
