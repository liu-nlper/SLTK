#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    训练并保存模型
"""
import os
import sys
import pickle
from time import time
import numpy as np
from optparse import OptionParser
from TorchNN.utils import read_pkl, SentenceDataUtil
from TorchNN.utils import is_interactive, parse_int_list
from TorchNN.layers import BiLSTMCRFModel, BiLSTMModel

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


op = OptionParser()
op.add_option('-f', dest='features', default=[0], type='str',
              action='callback', callback=parse_int_list, help='使用的特征列数')
op.add_option('--root_idx_train', dest='root_idx_train', default='./data/train_idx',
              type='str', help='训练数据索引根目录')
op.add_option('--root_idx_dev', dest='root_idx_dev', default='./data/dev_idx', type='str',
              help='开发数据索引根目录')
op.add_option('--rv', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--re', dest='root_embed', default='./res/embed', type='str', help='embed根目录')
op.add_option('--ml', dest='max_len', default=50, type='int', help='实例最大长度')
op.add_option('--mlc', dest='max_len_char', default=15, type='int', help='单词最大长度')
op.add_option('--ds', dest='dev_size', default=0.2, type='float', help='开发集占比')
op.add_option('--lstm', dest='lstm_units', default=256, type='int', help='LSTM单元数')
op.add_option('--ln', dest='layer_nums', default=1, type='int', help='LSTM层数')
op.add_option('--fd', dest='feature_dim', default=[64], type='str',
              action='callback', callback=parse_int_list, help='输入特征维度')
op.add_option('--dp', dest='dropout', default=0.5, type='float', help='dropout rate')
op.add_option('--lr', dest='learn_rate', default=0.002, type='float', help='learning rate')
op.add_option('--ne', dest='nb_epoch', default=100, type='int', help='迭代次数')
op.add_option('--mp', dest='max_patience', default=5,
              type='int', help='最大耐心值')
op.add_option('--rm', dest='root_model', default='./model/',
              type='str', help='模型根目录')
op.add_option('--bs', dest='batch_size', default=64, type='int', help='batch size')
op.add_option('-g', '--cuda', dest='cuda', action='store_true', default=False, help='是否使用GPU加速')
op.add_option('--nw', dest='nb_work', default=8, type='int', help='加载数据的线程数')
op.add_option('-c', '--char', dest='use_char_feature', action='store_true',
              default=False, help='是否使用字符特征')
op.add_option('--c_binary', dest='char_binary', action='store_true',
              default=False, help='字符特征是否one-hot编码')
op.add_option('--dc', dest='dim_char', default=16, type='int', help='字符特征维度')
op.add_option('--fs', dest='filter_sizes', default=[3], type='str',
              action='callback', callback=parse_int_list, help='卷积核尺寸(char feature)')
op.add_option('--fn', dest='filter_nums', default=[32], type='str',
              action='callback', callback=parse_int_list, help='卷积核数量(char feature)')
op.add_option('--device_ids', dest='device_ids', default=[0,1,2], type='str',
              action='callback', callback=parse_int_list, help='GPU编号列表')
op.add_option('--rg', dest='requires_grad', default='True', type='str', help='是否更新词向量')
op.add_option('--crf', dest='use_crf', action='store_true', default=False, help='是否使用CRF层')
op.add_option('--path_grid', dest='path_grid', default=None, type=str, help='是否保存模型参数')
op.add_option('--seed', dest='seed', default=1337, type='int', help='随机数种子')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)


class Arguments:

    def __init__(self, opts):

        self.opts = opts

        # 初始化数据参数
        self.init_data_args()

        # 初始化模型参数
        self.init_model_args()

    def init_data_args(self):
        self.seed = self.opts.seed
        self.root_idx_train = self.opts.root_idx_train
        self.root_idx_dev = self.opts.root_idx_dev
        self.path_num_train = os.path.join(self.opts.root_idx_train, 'nums.txt')
        self.max_len = self.opts.max_len
        self.root_voc = self.opts.root_voc
        self.features = self.opts.features
        self.feature2id_dict = dict()
        for feature_i in self.opts.features:
            path_f2id = os.path.join(self.root_voc, 'feature_{0}_2id.pkl'.format(feature_i))
            self.feature2id_dict[feature_i] = read_pkl(path_f2id)
        self.label2id_dict = read_pkl(os.path.join(self.root_voc, 'label2id.pkl'))
        self.feature2id_dict['label'] = self.label2id_dict
        self.has_label = True
        self.dev_size = self.opts.dev_size
        self.batch_size = self.opts.batch_size
        self.num_worker = self.opts.nb_work
        self.use_char_feature = self.opts.use_char_feature
        self.char_binary = self.opts.char_binary
        self.max_len_char = self.opts.max_len_char

    def init_model_args(self):
        path_embed = os.path.join(self.opts.root_embed, 'word2vec.pkl')
        pretrained_embed = None
        if os.path.exists(path_embed):
            pretrained_embed = read_pkl(path_embed)
        feature_size_dict = dict()
        for feature_name in self.feature2id_dict:
            feature_size_dict[feature_name] = len(self.feature2id_dict[feature_name]) + 1
        feature_dim_dict = dict()
        for i, feature_name in enumerate(self.features):
            if i < len(self.opts.feature_dim):
                feature_dim_dict[feature_name] = self.opts.feature_dim[i]
            else:
                feature_dim_dict[feature_name] = 32  # default value 32
        if pretrained_embed is not None:  # 以预训练向量维度为准
            feature_dim_dict[str(self.features[0])] = pretrained_embed.shape[-1]
        dropout_rate = self.opts.dropout
        dim_char = self.opts.dim_char
        filter_sizes = self.opts.filter_sizes
        filter_nums = self.opts.filter_nums
        self.use_cuda = self.opts.cuda
        self.device_ids = self.opts.device_ids
        self.multi_gpu = True if len(self.opts.device_ids) > 1 else False
        assert self.opts.requires_grad in ('False', 'True')
        requires_grad = True if self.opts.requires_grad == 'True' else False

        self.model_kwargs = {
            'features': self.features, 'lstm_units': self.opts.lstm_units, 'layer_nums': self.opts.layer_nums,
            'feature_size_dict': feature_size_dict, 'feature_dim_dict': feature_dim_dict,
            'pretrained_embed': pretrained_embed, 'dropout_rate': dropout_rate, 'max_len': self.max_len,
            'use_cuda': self.use_cuda, 'use_char_feature': self.use_char_feature, 'char_binary': self.char_binary,
            'dim_char': dim_char, 'filter_sizes': filter_sizes, 'filter_nums': filter_nums,
            'requires_grad': requires_grad, 'max_len_char': self.max_len_char, 'use_crf': self.opts.use_crf,
            'learn_rate': self.opts.learn_rate}

        self.learn_rate = self.opts.learn_rate
        self.nb_epoch = self.opts.nb_epoch
        self.max_patience = self.opts.max_patience
        self.root_model = self.opts.root_model
        self.use_crf = self.opts.use_crf


def init_dataset(arguments):
    """
    初始化数据集，包括训练集和开发集

    Args:
        arguments: Arguments

    Returns:
        data_loader_train: DataLoader
        data_loader_dev: DataLoader
    """
    args = arguments

    # 初始化训练数据
    dataset = SentenceDataUtil(
        args.path_num_train, args.root_idx_train, args.max_len, args.features,
        args.feature2id_dict, max_len_char=args.max_len_char,
        use_char_feature=args.use_char_feature, shuffle=False)

    # 若没有开发集，则从训练集中划分`dev_size`作为开发集
    if not os.path.exists(args.root_idx_dev):
        dataset_train, dataset_dev = dataset.split_train_and_dev(dev_size=args.dev_size)
    else:
        dataset_train = dataset.get_all_data()
        path_num_dev = os.path.join(args.root_idx_dev, 'nums.txt')
        dataset_dev = SentenceDataUtil(
            path_num_dev, args.root_idx_dev, args.max_len, args.features,
            args.feature2id_dict, shuffle=False, max_len_char=args.max_len_char,
            use_char_feature=args.use_char_feature).get_all_data()

    # 训练集和开发集
    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size*len(args.device_ids),
        shuffle=False, num_workers=args.num_worker)
    data_loader_dev = DataLoader(
        dataset_dev, batch_size=args.batch_size*len(args.device_ids),
        shuffle=False, num_workers=args.num_worker)

    return data_loader_train, data_loader_dev


def train(arguments, data_loader_train, data_loader_dev):
    """
    Args:
        arguments: Arguments
        data_loader_train: DataLoader
        data_loader_dev: DataLoader
    """
    torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(arguments.seed)
    # 初始化模型
    if arguments.use_crf:
        sl_model = BiLSTMCRFModel(arguments.model_kwargs)
    else:
        sl_model = BiLSTMModel(arguments.model_kwargs)
    print(sl_model)
    if arguments.use_cuda:
        # sl_model = sl_model.cuda(device=opts.device_ids[0])
        sl_model = sl_model.cuda()
    if arguments.multi_gpu:
        sl_model = nn.DataParallel(sl_model, arguments.device_ids).cuda(device=arguments.device_ids[0])
    parameters = filter(lambda p: p.requires_grad, sl_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=arguments.learn_rate)
    # optimizer = torch.optim.SGD(parameters, lr=opts.learn_rate, momentum=0.9, weight_decay=0)

    # 训练
    t0 = time()
    nb_epoch = arguments.nb_epoch
    max_patience = arguments.max_patience
    current_patience = 0
    root_model = arguments.root_model
    if not os.path.exists(root_model):
        os.makedirs(root_model)
    path_model = os.path.join(root_model, 'sl.model')
    best_dev_loss = sys.maxsize
    for epoch in range(nb_epoch):
        sys.stdout.write('epoch {0} / {1}: \r'.format(epoch+1, nb_epoch))
        total_loss, dev_loss = 0., 0.
        sl_model.train()
        current_count = 0
        data_loader_train.dataset.shuffle(arguments.seed)
        for i_batch, sample_batched in enumerate(data_loader_train):
            sl_model.zero_grad()
            for feature_name in sample_batched:
                if arguments.use_cuda:
                    sample_batched[str(feature_name)] = \
                        Variable(sample_batched[feature_name]).cuda()
                else:
                    sample_batched[str(feature_name)] = Variable(sample_batched[feature_name])

            lstm_feats = sl_model(**sample_batched)#.cuda(device=0)

            # note: origin module is sl_model.module
            model = sl_model if not arguments.multi_gpu else sl_model.module
            mask = None
            if arguments.use_crf:
                mask = sample_batched[str(arguments.features[0])] > 0
            loss = model.loss(lstm_feats, sample_batched['label'], mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]

            current_count += sample_batched[str(arguments.features[0])].size()[0]
            sys.stdout.write('epoch {0} / {1}: {2} / {3}\r'.format(
                epoch+1, nb_epoch, current_count, len(data_loader_train.dataset)))

        sys.stdout.write('epoch {0} / {1}: {2} / {3}\n'.format(
            epoch+1, nb_epoch, current_count, len(data_loader_train.dataset)))

        # 计算开发集loss
        sl_model.eval()
        for i_batch, sample_batched in enumerate(data_loader_dev):
            for feature_name in sample_batched:
                if arguments.use_cuda:
                    sample_batched[str(feature_name)] = Variable(sample_batched[feature_name]).cuda()
                else:
                    sample_batched[str(feature_name)] = Variable(sample_batched[feature_name])
            lstm_feats = sl_model(**sample_batched)

            # note: origin module is sl_model.module
            model = sl_model if not arguments.multi_gpu else sl_model.module
            mask = None
            if arguments.use_crf:
                mask = sample_batched[str(arguments.features[0])] > 0
            loss = model.loss(lstm_feats, sample_batched['label'], mask)
            dev_loss += loss.data[0]

        total_loss /= float(len(data_loader_train))
        dev_loss /= float(len(data_loader_dev))
        print('\ttrain loss: {:.4f}, dev loss: {:.4f}'.format(total_loss, dev_loss))

        # 根据开发集loss保存模型
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            # 保存模型
            model = sl_model if not arguments.multi_gpu else sl_model.module
            torch.save(model, path_model)
            print('\tmodel has saved to {0}!'.format(path_model))
            current_patience = 0
        else:
            current_patience += 1
            print('\tno improvement, current patience: {0} / {1}'.format(current_patience, max_patience))
            if max_patience <= current_patience:
                print('finished training! (early stopping, max patience: {0})'.format(max_patience))
                break
    duration = time() - t0
    print('finished training!')
    print('done in {:.1f}s!'.format(duration))

    # for parameters search
    return best_dev_loss


def run():
    arguments = Arguments(opts)
    data_loader_train, data_loader_dev = init_dataset(arguments)

    best_dev_loss = train(arguments, data_loader_train, data_loader_dev)

    if not opts.path_grid:
        return

    # 记录dev loss, for grid search
    with open(opts.path_grid, 'a', encoding='utf-8') as file_grid:
        argument_dict = dict()  # dict(arguments.model_kwargs)
        # 保留int, float, bool, list, dict类型的参数
        for key in arguments.model_kwargs:
            arg = arguments.model_kwargs[key]
            if isinstance(arg, int) or isinstance(arg, float) or isinstance(arg, bool) or \
               isinstance(arg, list) or isinstance(arg, dict):
                argument_dict.update({key: arguments.model_kwargs[key]})
        argument_dict.update({'dev_loss': best_dev_loss})
        file_grid.write('{0}\n'.format(argument_dict))


if __name__ == '__main__':
    run()
