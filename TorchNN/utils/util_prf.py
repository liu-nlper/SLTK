#!/usr/bin/env python
# -*- encoding: utf-8 -*-


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