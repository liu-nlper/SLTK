#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn

from ..functional import init_embedding


class CharFeature(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
            feature_size: int, 字符表的大小
            feature_dim: int, 字符embedding 维度
            require_grad: bool，char的embedding表是否需要更新

            filter_sizes: list(int), 卷积核尺寸
            filter_nums: list(int), 卷积核数量
        """
        super(CharFeature, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        # char embedding layer
        self.char_embedding = nn.Embedding(self.feature_size, self.feature_dim)
        init_embedding(self.char_embedding.weight)

        # cnn
        self.char_encoders = nn.ModuleList()
        for i, filter_size in enumerate(self.filter_sizes):
            f = nn.Conv3d(
                in_channels=1, out_channels=self.filter_nums[i], kernel_size=(1, filter_size, self.feature_dim))
            self.char_encoders.append(f)

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.feature_dim)

        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0])
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)

        # size=[bs, max_len, output_dim]
        char_conv_outputs = char_conv_outputs.transpose(-2, -1).contiguous()

        return char_conv_outputs


class WordFeature(nn.Module):

    def __init__(self, **kwargs):
        """
        Args:
             feature_names: list(str), 特征名称
             feature_size_dict: dict({str: int}), 特征名称到特征alphabet大小映射字典
             feature_dim_dict: dict({str: int}), 特征名称到特征维度映射字典

             require_grad_dict: dict({str: bool})，特征embedding表是否需要更新，特征名: bool
             pretrained_embed_dict: dict({str: np.array}): 特征的预训练embedding table
        """
        super(WordFeature, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        if not hasattr(self, 'require_grad_dict'):  # 默认需要更新
            self.require_grad_dict = dict()
            for feature_name in self.feature_names:
                self.require_grad_dict[feature_name] = True
        for feature_name in self.feature_names:
            if feature_name not in self.require_grad_dict:
                self.require_grad_dict[feature_name] = True

        if not hasattr(self, 'pretrained_embed_dict'):  # 默认随机初始化feature embedding
            self.pretrained_embed_dict = dict()
            for feature_name in self.feature_names:
                self.pretrained_embed_dict[feature_name] = None
        for feature_name in self.feature_names:
            if feature_name not in self.pretrained_embed_dict:
                self.pretrained_embed_dict[feature_name] = None

        # feature embedding layer
        self.feature_embedding_list = nn.ModuleList()
        for feature_name in self.feature_names:
            embed = nn.Embedding(self.feature_size_dict[feature_name], self.feature_dim_dict[feature_name])
            if self.pretrained_embed_dict[feature_name] is not None:  # 预训练向量
                # print('预训练:', feature_name)
                embed.weight.data.copy_(torch.from_numpy(self.pretrained_embed_dict[feature_name]))
            else:  # 随机初始化
                # print('随机初始化:', feature_name)
                init_embedding(embed.weight)
            # 是否需要根据embedding的权重
            embed.weight.requires_grad = self.require_grad_dict[feature_name]
            self.feature_embedding_list.append(embed)

    def forward(self, **input_dict):
        """
        Args:
            input_dict: dict({str: LongTensor})

        Returns:
            embed_outputs: 3D tensor, [bs, max_len, input_size]
        """
        embed_outputs = []
        for i, feature_name in enumerate(self.feature_names):
            embed_outputs.append(self.feature_embedding_list[i](input_dict[feature_name]))
        embed_outputs = torch.cat(embed_outputs, dim=2)  # size=[bs, max_len, input_size]

        return embed_outputs
