#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    模型训练类.
"""
import sys
import numpy as np

import torch


class SLTrainer(object):

    def __init__(self, **kwargs):
        """
        Args of data:
            data_iter_train: 训练数据迭代器
            data_iter_dev: 开发数据迭代器
            feature_names: list(str), 特征名称, 没有`label`和`char`
            use_char: bool, 是否使用char feature
            max_len_char: int, 单词最大长度

        Args of train:
            model: 初始化之后的模型
            optimizer: model arguments optimizer
            lr_decay: float, 学习率衰减率
            learning_rate: float, 初始学习率
            path_save_model: str, 模型保存的路径

            nb_epoch: int, 迭代次数上限
            max_patience: int, 开发集上连续mp次没有提升即停止训练
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def _get_inputs(self, feed_dict, use_cuda=True):
        feed_tensor_dict = dict()
        for feature_name in self.feature_names:
            tensor = self.tensor_from_numpy(  # [bs, max_len]
                feed_dict[feature_name], use_cuda=use_cuda)
            feed_tensor_dict[feature_name] = tensor
        if self.use_char:  # max_len_char
            char_tensor = self.tensor_from_numpy(
                feed_dict['char'], use_cuda=self.model.use_cuda)
            max_len = feed_dict[self.feature_names[0]].shape[-1]
            char_tensor = char_tensor.view(-1, max_len, self.max_len_char)
            feed_tensor_dict['char'] = char_tensor
        return feed_tensor_dict

    def fit(self):
        """训练模型
        """
        best_dev_loss = 1.e8
        current_patience = 0
        for epoch in range(self.nb_epoch):
            train_loss, dev_loss = 0., 0.
            self.model.train()
            if self.lr_decay != 0.:
                self.optimizer = self.decay_learning_rate(epoch, self.learning_rate)
            for i, feed_dict in enumerate(self.data_iter_train):
                self.optimizer.zero_grad()
                feed_tensor_dict = self._get_inputs(feed_dict, self.model.use_cuda)

                labels = self.tensor_from_numpy(feed_dict['label'], 'long', self.model.use_cuda)

                # mask
                mask = feed_tensor_dict[str(self.feature_names[0])] > 0

                logits = self.model(**feed_tensor_dict)
                loss = self.model.loss(logits, mask, labels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                sys.stdout.write('Epoch {0} / {1}: {2} / {3}\r'.format(
                    epoch+1, self.nb_epoch, self.data_iter_train.iter_variable, self.data_iter_train.data_count))
            sys.stdout.write('Epoch {0} / {1}: {2} / {3}\n'.format(
                epoch+1, self.nb_epoch, self.data_iter_train.data_count, self.data_iter_train.data_count))

            # 计算开发集loss
            self.model.eval()
            # dev_labels_pred, dev_labels_gold = [], []
            for feed_dict in self.data_iter_dev:
                feed_tensor_dict = self._get_inputs(feed_dict, self.model.use_cuda)

                labels = self.tensor_from_numpy(feed_dict['label'], 'long', self.model.use_cuda)

                logits = self.model(**feed_tensor_dict)
                # mask
                mask = feed_tensor_dict[str(self.feature_names[0])] > 0
                loss = self.model.loss(logits, mask, labels)
                dev_loss += loss.item()

            print('\ttrain loss: {0}, dev loss: {1}'.format(train_loss, dev_loss))

            # 判断是否需要保存模型
            if dev_loss < best_dev_loss:
                current_patience = 0
                best_dev_loss = dev_loss
                # 保存模型
                self.save_model()
                print('\tmodel has saved to {0}!'.format(self.path_save_model))
            else:
                current_patience += 1
                print('\tno improvement, current patience: {0} / {1}'.format(
                    current_patience, self.max_patience))
                if self.max_patience <= current_patience:
                    print('finished training! (early stopping, max patience: {0})'.format(self.max_patience))
                    return
        print('finished training!')

    def predict(self, data_iter, has_label=False):
        """预测
        Args:
            data_iter: 数据迭代器
            has_label: bool, 是否带有label

        Returns:
            labels: list of int
        """
        labels_pred, labels_gold = [], []
        for feed_dict in data_iter:
            if has_label:
                labels_gold_batch = np.array(feed_dict['label']).astype(np.int32).tolist()
                labels_gold.extend(labels_gold_batch)
            feed_tensor_dict = self._get_inputs(feed_dict, self.model.use_cuda)

            logits = self.model(**feed_tensor_dict)
            # mask
            mask = feed_tensor_dict[str(self.feature_names[0])] > 0
            actual_lens = torch.sum(feed_tensor_dict[self.feature_names[0]] > 0, dim=1).int()
            labels_batch = self.model.predict(logits, actual_lens, mask)
            labels_pred.extend(labels_batch)
        if has_label:
            return labels_gold, labels_pred
        return labels_pred

    def decay_learning_rate(self, epoch, init_lr):
        """衰减学习率

        Args:
            epoch: int, 迭代次数
            init_lr: 初始学习率
        """
        lr = init_lr / (1+self.lr_decay*epoch)
        print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    @staticmethod
    def tensor_from_numpy(data, dtype='long', use_cuda=True):
        """将numpy转换为tensor
        Args:
            data: numpy
            dtype: long or float
            use_cuda: bool
        """
        assert dtype in ('long', 'float')
        if dtype == 'long':
            data = torch.from_numpy(data).long()
        else:
            data = torch.from_numpy(data).float()
        if use_cuda:
            data = data.cuda()
        return data

    def save_model(self):
        """保存模型
        """
        torch.save(self.model.state_dict(), self.path_save_model)

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.data_iter_train.batch_size = batch_size
        self.data_iter_dev.batch_size = batch_size

    def set_max_patience(self, max_patience):
        self.max_patience = max_patience

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate
