#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import sys
import codecs
import torch

from ..utils import read_conllu


class Inference(object):

    def __init__(self, **kwargs):
        """
        Args:
            model: SLModel
            data_iter: DataIter
            path_conllu: str, conllu格式的文件路径
            path_result: str, 预测结果存放路径
            label2id_dict: dict({str: int})
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        # id2label dict
        self.id2label_dict = dict()
        for label in self.label2id_dict:
            self.id2label_dict[self.label2id_dict[label]] = label

    def _get_inputs(self, feed_dict, use_cuda=True):
        feed_tensor_dict = dict()
        for feature_name in self.model.feature_names:
            tensor = self.tensor_from_numpy(  # [bs, max_len]
                feed_dict[feature_name], use_cuda=use_cuda)
            feed_tensor_dict[feature_name] = tensor
        if self.model.use_char:  # max_len_char
            char_tensor = self.tensor_from_numpy(
                feed_dict['char'], use_cuda=self.model.use_cuda)
            max_len = feed_dict[self.model.feature_names[0]].shape[-1]
            char_tensor = char_tensor.view(-1, max_len, self.data_iter.char_max_len)
            feed_tensor_dict['char'] = char_tensor
        return feed_tensor_dict

    def infer(self):
        """预测
        Returns:
            labels: list of int
        """
        self.model.eval()
        labels_pred = []
        for feed_dict in self.data_iter:
            feed_tensor_dict = self._get_inputs(feed_dict, self.model.use_cuda)

            logits = self.model(**feed_tensor_dict)
            # mask
            mask = feed_tensor_dict[str(self.feature_names[0])] > 0
            actual_lens = torch.sum(feed_tensor_dict[self.feature_names[0]] > 0, dim=1).int()
            labels_batch = self.model.predict(logits, actual_lens, mask)
            labels_pred.extend(labels_batch)
            sys.stdout.write('sentence: {0} / {1}\r'.format(self.data_iter.iter_variable, self.data_iter.data_count))
        sys.stdout.write('sentence: {0} / {1}\n'.format(self.data_iter.data_count, self.data_iter.data_count))

        return labels_pred

    def infer2file(self):
        """预测，将结果写入文件
        """
        self.model.eval()
        file_result = codecs.open(self.path_result, 'w', encoding='utf-8')
        conllu_reader = read_conllu(self.path_conllu, zip_format=False)
        for feed_dict in self.data_iter:
            feed_tensor_dict = self._get_inputs(feed_dict, self.model.use_cuda)

            logits = self.model(**feed_tensor_dict)
            # mask
            mask = feed_tensor_dict[str(self.model.feature_names[0])] > 0
            actual_lens = torch.sum(feed_tensor_dict[self.model.feature_names[0]] > 0, dim=1).int()
            label_ids_batch = self.model.predict(logits, actual_lens, mask)
            labels_batch = self.id2label(label_ids_batch)  # list(list(int))

            # write to file
            batch_size = len(labels_batch)
            for i in range(batch_size):
                feature_items = conllu_reader.__next__()
                sent_len = len(feature_items)  # 句子实际长度
                labels = labels_batch[i]
                if len(labels) < sent_len:  # 补全为`O`
                    labels = labels + ['O'] * (sent_len-len(labels))
                for j in range(sent_len):
                    file_result.write('{0} {1}\n'.format(' '.join(feature_items[j]), labels[j]))
                file_result.write('\n')

            sys.stdout.write('sentence: {0} / {1}\r'.format(self.data_iter.iter_variable, self.data_iter.data_count))
            sys.stdout.flush()
        sys.stdout.write('sentence: {0} / {1}\n'.format(self.data_iter.data_count, self.data_iter.data_count))
        sys.stdout.flush()

        file_result.close()

    def id2label(self, label_ids_array):
        """将label ids转为label
        Args:
            label_ids_array: list(np.array)

        Returns:
            labels: list(list(str))
        """
        labels = []
        for label_array in label_ids_array:
            temp = []
            for idx in label_array:
                temp.append(self.id2label_dict[idx])
            labels.append(temp)
        return labels

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

