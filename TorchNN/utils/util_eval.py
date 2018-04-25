#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import re
from tqdm import tqdm


def P_R_F(correct, preci, test):
    """
    Args:
        correct: 正确的个数
        preci: 预测的个数
        test: 测试集中实体总数
    Return:
        p, r, f
    """
    if preci == 0:
        precisity = 0
    else:
        precisity = 100 * correct / preci
    if test == 0:
        recall = 0
    else:
        recall = 100 * correct / test
    if precisity + recall == 0:
        f_measure = 0
    else:
        f_measure = (2 * precisity * recall) / (precisity + recall)
    return precisity, recall, f_measure


def extract_entity(sentence, column):
    """
    抽出句中实体位置实体
    Args:
        sentence: lines
        column: 第几列
    Return:

    """
    flag = False
    start, s = 0, 0
    loca_indice = []
    word_lines = sentence.split('\n')
    for i in range(len(word_lines)):
        if not word_lines[i]:
            continue
        word_info = word_lines[i].split('\t')
        # 实体标签
        if word_info[column][0] == 'S':
            loca_indice.append([i, i])

        if word_info[column][0] == 'B':
            start = i
            flag = True
        if word_info[column][0] == 'E':
            end = i
            if flag:
                loca_indice.append([start, end])
                flag = False
            else:
                flag = False
        if word_info[column][0] == 'O':
            flag = False
    return loca_indice


def extract_indice_entity(pmid, docu_list, text_dict, abbre_dict):
    """
    字符串匹配文本，抽出位置、实体信息
    """
    context = ''.join(text_dict[pmid])
    entity_list = []
    unfind_entity = []
    last_length = 0
    for i in range(len(docu_list)):
        word, tag = docu_list[i][:]
        start = context.find(word)
        # if context.find(word) == '-1':
        end = start + len(word)
        if tag[0] == 'S':
            entity_list.append([start+last_length, end+last_length])
        if tag[0] == 'B':
            if start == -1:
                unfind_entity.append(word)
            else:
                s = start + last_length
        if tag[0] == 'I' and start == -1:
            unfind_entity.append(word)
        if tag[0] == 'E':
            if start == -1:
                unfind_entity.append(word)
                entity = ' '.join(unfind_entity)
                print([pmid, entity])
                unfind_entity = []
            else:
                e = end + last_length
            entity_list.append([s, e])
        last_length += end
        context = context[end:]
    return entity_list


def generate_dict(lines):
    """
    给出预测文件所有lines，返回dict(){'998881':[[12,14],'hello', ...]}
    """
    # 收集predict entity到dict
    entity_dict = dict()
    pre_id = ''
    entity_list = []
    # print(lines)
    for line in tqdm(lines):
        if not line:
            continue
        info = line.split('\t')
        present_id = info[0]
        # print(line)
        start = int(info[1])
        end = int(info[2])
        if len(info) < 6:
            Norm_ID = '0'
        else:
            Norm_ID = info[-1]
        if pre_id and pre_id != present_id:
            entity_dict[pre_id] = entity_list
            entity_list = []
            entity_list.append([[start, end], info[3], Norm_ID])
            pre_id = present_id
        else:
            pre_id = present_id
            entity_list.append([[start, end], info[3], Norm_ID])
    entity_dict[pre_id] = entity_list
    return entity_dict
