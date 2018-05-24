#!/usr/bin/env python
# -*- encoding: utf-8 -*-


def normalize_word(word):
    new_word = ''
    for c in word:
        if c.isdigit():
            new_word += '0'
        else:
            new_word += c
    return new_word
