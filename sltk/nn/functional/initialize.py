#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn


def init_cnn_weight(cnn_layer, seed=1337):
    """初始化cnn层权重
    Args:
        cnn_layer: weight.size() == [nb_filter, in_channels, [kernel_size]]
        seed: int
    """
    filter_nums = cnn_layer.weight.size(0)
    kernel_size = cnn_layer.weight.size()[2:]
    scope = np.sqrt(2. / (filter_nums * np.prod(kernel_size)))
    torch.manual_seed(seed)
    nn.init.normal_(cnn_layer.weight, -scope, scope)
    cnn_layer.bias.data.zero_()


def init_lstm_weight(lstm, num_layer=1, seed=1337):
    """初始化lstm权重
    Args:
        lstm: torch.nn.LSTM
        num_layer: int, lstm层数
        seed: int
    """
    for i in range(num_layer):
        weight_h = getattr(lstm, 'weight_hh_l{0}'.format(i))
        scope = np.sqrt(6.0 / (weight_h.size(0)/4. + weight_h.size(1)))
        torch.manual_seed(seed)
        nn.init.uniform_(getattr(lstm, 'weight_hh_l{0}'.format(i)), -scope, scope)

        weight_i = getattr(lstm, 'weight_ih_l{0}'.format(i))
        scope = np.sqrt(6.0 / (weight_i.size(0)/4. + weight_i.size(1)))
        torch.manual_seed(seed)
        nn.init.uniform_(getattr(lstm, 'weight_ih_l{0}'.format(i)), -scope, scope)

    if lstm.bias:
        for i in range(num_layer):
            weight_h = getattr(lstm, 'bias_hh_l{0}'.format(i))
            weight_h.data.zero_()
            weight_h.data[lstm.hidden_size: 2*lstm.hidden_size] = 1
            weight_i = getattr(lstm, 'bias_ih_l{0}'.format(i))
            weight_i.data.zero_()
            weight_i.data[lstm.hidden_size: 2*lstm.hidden_size] = 1


def init_linear(input_linear, seed=1337):
    """初始化全连接层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_embedding(input_embedding, seed=1337):
    """初始化embedding层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)
