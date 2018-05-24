#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            rnn_unit_type: str, options: ['rnn', 'lstm', 'gru']
            input_dim: int, rnn输入维度
            num_rnn_units: int, rnn单元数
            num_layers: int, 层数
            bi_flag: bool, 是否双向, default is True
        """
        super(RNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        if not hasattr(self, 'bi_flag'):
            self.__setattr__('bi_flag', True)

        if self.rnn_unit_type == 'rnn':
            self.rnn = nn.RNN(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)
        elif self.rnn_unit_type == 'gru':
            self.rnn = nn.GRU(self.input_dim, self.num_rnn_units, self.num_layers, bidirectional=self.bi_flag)

    def forward(self, feats):
        """
        Args:
             feats: 3D tensor, shape=[bs, max_len, input_dim]

        Returns:
            rnn_outputs: 3D tensor, shape=[bs, max_len, self.rnn_unit_num]
        """
        rnn_outputs, _ = self.rnn(feats)
        return rnn_outputs
