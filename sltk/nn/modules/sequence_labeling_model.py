#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    Sequence Labeling Model.
"""
import torch
import torch.nn as nn

from .rnn import RNN
from .crf import CRF
from .feature import CharFeature, WordFeature


class SLModel(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            feature_names: list(str), 特征名称, 不包括`label`和`char`

            feature_size_dict: dict({str: int}), 特征表大小字典
            feature_dim_dict: dict({str: int}), 输入特征dim字典
            pretrained_embed_dict: dict({str: np.array})
            require_grad_dict: bool, 是否更新feature embedding的权重

            # char parameters
            use_char: bool, 是否使用字符特征, default is False
            filter_sizes: list(int), 卷积核尺寸, default is [3]
            filter_nums: list(int), 卷积核数量, default is [32]

            # rnn parameters
            rnn_unit_type: str, options: ['rnn', 'lstm', 'gru']
            num_rnn_units: int, rnn单元数
            num_layers: int, 层数
            bi_flag: bool, 是否双向, default is True

            use_crf: bool, 是否使用crf层

            dropout_rate: float, dropout rate

            average_batch: bool, 是否对batch的loss做平均
            use_cuda: bool
        """
        super(SLModel, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        # word level feature layer
        self.word_feature_layer = WordFeature(
            feature_names=self.feature_names, feature_size_dict=self.feature_size_dict,
            feature_dim_dict=self.feature_dim_dict, require_grad_dict=self.require_grad_dict,
            pretrained_embed_dict=self.pretrained_embed_dict)
        rnn_input_dim = 0
        for name in self.feature_names:
            rnn_input_dim += self.feature_dim_dict[name]

        # char level feature layer
        if self.use_char:
            self.char_feature_layer = CharFeature(
                feature_size=self.feature_size_dict['char'], feature_dim=self.feature_dim_dict['char'],
                require_grad=self.require_grad_dict['char'], filter_sizes=self.filter_sizes,
                filter_nums=self.filter_nums)
            rnn_input_dim += sum(self.filter_nums)

        # feature dropout
        self.dropout_feature = nn.Dropout(self.dropout_rate)

        # rnn layer
        self.rnn_layer = RNN(
            rnn_unit_type=self.rnn_unit_type, input_dim=rnn_input_dim, num_rnn_units=self.num_rnn_units,
            num_layers=self.num_layers, bi_flag=self.bi_flag)

        # rnn dropout
        self.dropout_rnn = nn.Dropout(self.dropout_rate)

        # crf layer
        self.target_size = self.feature_size_dict['label']
        args_crf = dict({'target_size': self.target_size, 'use_cuda': self.use_cuda})
        args_crf['average_batch'] = self.average_batch
        if self.use_crf:
            self.crf_layer = CRF(**args_crf)

        # dense layer
        hidden_input_dim = self.num_rnn_units * 2 if self.bi_flag else self.num_rnn_units
        target_size = self.target_size + 2 if self.use_crf else self.target_size
        self.hidden2tag = nn.Linear(hidden_input_dim, target_size)

        # loss
        if not self.use_crf:
            self.loss_function = nn.CrossEntropyLoss(ignore_index=0, size_average=False)
        else:
            self.loss_function = self.crf_layer.neg_log_likelihood_loss

    def loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        if not self.use_crf:
            batch_size, max_len = feats.size(0), feats.size(1)
            lstm_feats = feats.view(batch_size * max_len, -1)
            tags = tags.view(-1)
            return self.loss_function(lstm_feats, tags)
        else:
            loss_value = self.loss_function(feats, mask, tags)
        if self.average_batch:
            batch_size = feats.size(0)
            loss_value /= float(batch_size)
        return loss_value

    def forward(self, **feed_dict):
        """
        Args:
             inputs: list
        """
        batch_size = feed_dict[self.feature_names[0]].size(0)
        max_len = feed_dict[self.feature_names[0]].size(1)

        # word level feature
        word_feed_dict = {}
        for i, feature_name in enumerate(self.feature_names):
            word_feed_dict[feature_name] = feed_dict[feature_name]
        word_feature = self.word_feature_layer(**word_feed_dict)

        # char level feature
        if self.use_char:
            char_feature = self.char_feature_layer(feed_dict['char'])
            word_feature = torch.cat([word_feature, char_feature], 2)

        word_feature = self.dropout_feature(word_feature)
        word_feature = torch.transpose(word_feature, 1, 0)  # size=[max_len, bs, input_size]

        # rnn layer
        rnn_outputs = self.rnn_layer(word_feature)
        rnn_outputs = rnn_outputs.transpose(1, 0).contiguous()  # [bs, max_len, lstm_units]

        rnn_outputs = self.dropout_rnn(rnn_outputs.view(-1, rnn_outputs.size(-1)))
        rnn_feats = self.hidden2tag(rnn_outputs)

        return rnn_feats.view(batch_size, max_len, -1)

    def predict(self, rnn_outputs, actual_lens, mask=None):
        batch_size = rnn_outputs.size(0)
        tags_list = []

        if not self.use_crf:
            _, arg_max = torch.max(rnn_outputs, dim=2)  # [batch_size, max_len]
            for i in range(batch_size):
                tags_list.append(arg_max[i].cpu().data.numpy()[:actual_lens.data[i]])
        else:
            path_score, best_paths = self.crf_layer(rnn_outputs, mask)
            for i in range(batch_size):
                tags_list.append(best_paths[i].cpu().data.numpy()[:actual_lens.data[i]])

        return tags_list
