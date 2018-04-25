#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import sys
import codecs
from time import time
from optparse import OptionParser
from TorchNN.utils import read_pkl, SentenceDataUtil
from TorchNN.utils import is_interactive, parse_int_list

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from TorchNN.utils import extract_entity, P_R_F


op = OptionParser()
op.add_option('--ri', '--root_idx', dest='root_idx', default='./data/test_idx', type='str', help='数据索引根目录')
op.add_option('--rv', '--root_voc', dest='root_voc', default='./res/voc', type='str', help='字典根目录')
op.add_option('--pm', '--path_model', dest='path_model', default='./model/sl.model',
              type='str', help='模型路径')
op.add_option('--ml', dest='max_len', default=50, type='int', help='实例最大长度')
op.add_option('--bs', '--batch_size', dest='batch_size', default=64, type='int', help='batch size')
op.add_option('-g', '--cuda', dest='cuda', action='store_true', default=False, help='是否使用GPU加速')
op.add_option('--nw', dest='nb_work', default=8, type='int', help='加载数据的线程数')
op.add_option('--device_ids', dest='device_ids', default=[0,1,2], type='str',
              action='callback', callback=parse_int_list, help='GPU编号列表')
op.add_option('-o', '--output', dest='output', default='./data/result.txt',
              type='str', help='预测结果存放路径')
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)



def init_model(opts):
    """
    初始化模型

    Return:
        sl_model: nn.Module or DataParallel
    """
    path_model = opts.path_model
    sl_model = torch.load(path_model)
    sl_model.set_use_cuda(opts.cuda)
    sl_model.max_len = opts.max_len
    sl_model.eval()
    if opts.cuda:
        sl_model = sl_model.cuda()
    else:
        sl_model = sl_model.cpu()
    if len(opts.device_ids)>1 and opts.cuda:
        sl_model = nn.DataParallel(sl_model, opts.device_ids).cuda(device=opts.device_ids[0])

    return sl_model


def init_dataset(opts, features, max_len, use_char_feature, max_len_char):
    """
    初始化数据参数

    Args:
        opts:
        features: list of int
        max_len: int
        use_char_feature: bool
        max_len_char: int

    Return:
        data_loader_test: DataLoader
    """
    root_idx = opts.root_idx
    path_num = os.path.join(root_idx, 'nums.txt')
    root_voc = opts.root_voc
    feature2id_dict = dict()
    for feature_i in features:
        path_f2id = os.path.join(root_voc, 'feature_{0}_2id.pkl'.format(feature_i))
        feature2id_dict[feature_i] = read_pkl(path_f2id)
    label2id_dict = read_pkl(os.path.join(root_voc, 'label2id.pkl'))
    has_label = False
    batch_size = opts.batch_size
    use_cuda = opts.cuda
    num_worker = opts.nb_work
    path_result = opts.output

    # 初始化数据
    dataset = SentenceDataUtil(
        path_num, root_idx, max_len, features, feature2id_dict,
        max_len_char=max_len_char, use_char_feature=use_char_feature, shuffle=False)
    dataset_test = dataset.get_all_data()
    data_loader_test = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return data_loader_test, label2id_dict


def test(sl_model, data_loader_test, label2id_dict, path_result, opts):
    """
    测试

    Args:
        data_loader_test: DataLoader
        label2id_dict: dict
    """
    t0 = time()
    # 测试
    label2id_dict_rev = dict()
    for k, v in label2id_dict.items():
        label2id_dict_rev[v] = k
    label2id_dict_rev[0] = 'O'
    file_result = codecs.open(path_result, 'w', encoding='utf-8')
    current_count, total_count = 0, len(data_loader_test.dataset)
    model = sl_model if len(opts.device_ids) == 1 else sl_model.module
    features = model.features
    file_num_idx = 0
    for i_batch, sample_batched in enumerate(data_loader_test):
        current_count += sample_batched[str(features[0])].size()[0]
        sys.stdout.write('{0} / {1}\r'.format(current_count, total_count))
        for feature_name in sample_batched:
            if opts.cuda:
                sample_batched[str(feature_name)] = Variable(sample_batched[feature_name]).cuda()
            else:
                sample_batched[str(feature_name)] = Variable(sample_batched[feature_name])

        lstm_feats = sl_model(**sample_batched)
        mask = None
        if model.use_crf:
            mask = sample_batched[str(features[0])] > 0
        actual_lens = torch.sum(sample_batched[str(features[0])]>0, dim=1).int()

        targets_list = model.predict(lstm_feats, actual_lens, mask)
        for targets in targets_list:
            path_test_i = os.path.join(opts.root_idx, '{0}.txt'.format(file_num_idx))
            file_test_i = codecs.open(path_test_i, 'r', encoding='utf-8')
            lines = file_test_i.read().strip().split('\n')
            file_test_i.close()
            file_num_idx += 1

            targets = list(map(lambda d: label2id_dict_rev[d], targets))
            for i, line in enumerate(lines):
                target = targets[i] if i < len(targets) else 'O'
                file_result.write('{0}\t{1}\n'.format(line.strip(), target))
            file_result.write('\n')
    sys.stdout.write('{0} / {1}\n'.format(current_count, total_count))
    file_result.close()
    print('done in {:.1f}s!'.format(time()-t0))


if __name__ == '__main__':
    sl_model = init_model(opts)

    model = sl_model if len(opts.device_ids) == 1 else sl_model.module
    features = model.features
    max_len = model.max_len
    use_char_feature = model.use_char_feature
    max_len_char = model.max_len_char
    data_loader_test, label2id_dict = init_dataset(
        opts, features, max_len, use_char_feature, max_len_char)

    path_result = opts.output
    test(sl_model, data_loader_test, label2id_dict, path_result, opts)
