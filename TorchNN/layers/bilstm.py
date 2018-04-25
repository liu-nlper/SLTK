#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    Sequence Labeling Model
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from TorchNN.utils import init_linear, init_embedding, \
    init_cnn_weight, init_lstm_weight


class BiLSTMModel(nn.Module):

    def __init__(self, args):
        """
        Args:
            feature_size_dict: dict, 特征表大小字典
            feature_dim_dict: dict, 输入特征dim字典
            pretrained_embed: np.array, default is None
            dropout_rate: float, dropout rate
        """
        super(BiLSTMModel, self).__init__()
        for k, v in args.items():
            self.__setattr__(k, v)
        if not hasattr(self, 'dropout_rate'):
            self.__setattr__('dropout_rate', '0.5')
        if not hasattr(self, 'seed'):
            self.__setattr__('seed', 1337)
        if not hasattr(self, 'dim_char'):
            self.__setattr__('seed', 16)
        if not hasattr(self, 'filter_sizes'):
            self.__setattr__('filter_sizes', [3])
        if not hasattr(self, 'filter_nums'):
            self.__setattr__('filter_nums', [32])
        if not hasattr(self, 'requires_grad'):
            self.__setattr__('requires_grad', True)

        # feature embedding layer
        self.embedding_list = nn.ModuleList()
        lstm_input_dim = 0
        for i, feature_name in enumerate(self.features):
            embed = nn.Embedding(
                self.feature_size_dict[feature_name], self.feature_dim_dict[feature_name])
            if i == 0 and not self.requires_grad:
                embed.weight.requires_grad = False
            self.embedding_list.append(embed)
            if feature_name != 'label':
                lstm_input_dim += self.feature_dim_dict[feature_name]
        if hasattr(self, 'pretrained_embed') and self.pretrained_embed is not None:
            self.embedding_list[0].weight.data.copy_(torch.from_numpy(self.pretrained_embed))
            #if self.use_cuda:
            #    self.embedding_list[0].weight.data = self.embedding_list[0].weight.data.cuda()

        # char embedding layer
        if self.use_char_feature:
            from string import ascii_letters, digits
            self.embedding_char = nn.Embedding(len(ascii_letters+digits)+2, self.dim_char)
            self.char_encoders = nn.ModuleList()
            for i, filter_size in enumerate(self.filter_sizes):
                f = nn.Conv3d(
                    in_channels=1, out_channels=self.filter_nums[i], kernel_size=(1, filter_size, self.dim_char))
                #f = f.cuda() if self.use_cuda else f
                self.char_encoders.append(f)
            lstm_input_dim += sum(self.filter_nums)

        # lstm layer
        self.lstm = nn.LSTM(
            lstm_input_dim, self.lstm_units, num_layers=self.layer_nums,
            bidirectional=True, dropout=0.2)

        # dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)

        # dense layer
        self.hidden2tag = nn.Linear(self.lstm_units*2, self.feature_size_dict['label'])

        self._init_weight()

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

    def get_lstm_feats(self, input_dict, batch_size):
        """
        Returns:
            tag_scores: size=[batch_size * max_len, nb_classes]
        """
        # concat inputs
        inputs = []
        for i, feature_name in enumerate(self.features):
            inputs.append(self.embedding_list[i](input_dict[str(feature_name)]))
        inputs = torch.cat(inputs, dim=2)  # size=[batch_size, max_len, input_size]

        # char feature layer
        if self.use_char_feature:
            char_feature_input = input_dict['char_feature']  # size=[bs, max_len, max_len_char]
            self.max_len, self.max_len_char = char_feature_input.size(1), char_feature_input.size(2)
            char_feature_input = char_feature_input.view(-1, self.max_len*self.max_len_char)
            char_feature_embed = self.embedding_char(char_feature_input)
            # size=[bs, max_len, max_len_char, dim_char]
            char_feature_embed = char_feature_embed.view(-1, 1, self.max_len, self.max_len_char, self.dim_char)

            # conv
            char_conv_outputs = []
            for char_encoder in self.char_encoders:
                conv_output = char_encoder(char_feature_embed)
                pool_output = torch.squeeze(torch.max(conv_output, -2)[0])
                char_conv_outputs.append(pool_output)
            char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
            # size=[bs, max_len, output_dim]
            char_conv_outputs = char_conv_outputs.transpose(-2, -1).contiguous()

            # concat with other features
            inputs = torch.cat([inputs, char_conv_outputs], -1)

        inputs = torch.transpose(inputs, 1, 0)  # size=[max_len, batch_size, input_size]

        self.lstm.flatten_parameters()
        lstm_output, _ = self.lstm(inputs)
        lstm_output = lstm_output.transpose(1, 0).contiguous()  # [batch_size, max_len, lstm_units]

        lstm_output = self.dropout(lstm_output.view(-1, self.lstm_units*2))

        # [batch_size * max_len, target_size]
        lstm_feats = self.hidden2tag(lstm_output)

        return lstm_feats.view(batch_size, self.max_len, -1)

    def forward(self, **input_dict):
        """
        Args:
            inputs: autograd.Variable, size=[batch_size, max_len]
        """
        batch_size = input_dict[str(self.features[0])].size()[0]

        return self.get_lstm_feats(input_dict, batch_size)

    def loss(self, lstm_feats, gold_tag, mask=None):
        """
        计算loss

        Args:
            lstm_feats:  size=(batch_size, max_len, tag_size)
            gold_tag: size=(batch_size, max_len)
        """
        batch_size, max_len = lstm_feats.size(0), lstm_feats.size(1)
        lstm_feats = lstm_feats.view(batch_size*max_len, -1)
        gold_tag = gold_tag.view(-1)
        return self.loss_function(lstm_feats, gold_tag)

    def predict(self, lstm_feats, actual_lens, mask=None):
        """
        预测标签
        """
        batch_size = lstm_feats.size(0)
        _, arg_max = torch.max(lstm_feats, dim=2)  # [batch_size, max_len]
        tags_list = []
        for i in range(batch_size):
            tags_list.append(arg_max[i].cpu().data.numpy()[:actual_lens.data[i]])

        return tags_list

    def _init_weight(self):
        """
        初始化参数权重
        """
        # embedding layer
        if hasattr(self, 'pretrained_embed') and self.pretrained_embed is None:
            init_embedding(self.embedding_list[0].weight, seed=self.seed)
            #if self.use_cuda:
            #    self.embedding_list[0].weight.data = self.embedding_list[0].weight.data.cuda()
        for i in range(1, len(self.features)):
            init_embedding(
                self.embedding_list[i].weight, seed=self.seed)
            #if self.use_cuda:
            #    self.embedding_list[i].weight.data = self.embedding_list[i].weight.data.cuda()

        # conv layer
        if self.use_char_feature:
            init_embedding(self.embedding_char.weight, seed=self.seed)
            for char_encoder in self.char_encoders:
                init_cnn_weight(char_encoder)

        # lstm layer 
        init_lstm_weight(self.lstm, seed=self.seed)

        # dense layer
        init_linear(self.hidden2tag, seed=self.seed)


    def set_use_cuda(self, use_cuda):
        self.use_cuda = use_cuda
