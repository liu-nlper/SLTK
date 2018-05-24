#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    将BIO标注转换为BIESO
        
    Usage:
    
        python3 bio2bieo.py -i input.txt -o output.txt
"""
import re
import sys
import codecs
from optparse import OptionParser


def read_conllu(path, zip_format=True):
    """读取conllu格式文件
    Args:
         path: str

    yield:
        list(list)
    """
    pattern_space = re.compile('\s+')
    feature_items = []
    file_data = codecs.open(path, 'r', encoding='utf-8')
    line = file_data.readline()
    line_idx = 1
    while line:
        line = line.strip()
        if not line:
            # 判断是否存在多个空行
            if not feature_items:
                print('存在多个空行！`{0}` line: {1}'.format(path, line_idx))
                exit()

            # 处理上一个实例
            if zip_format:
                yield list(zip(*feature_items))
            else:
                yield feature_items

            line = file_data.readline()
            line_idx += 1
            feature_items = []
        else:
            # 记录特征
            items = pattern_space.split(line)
            feature_items.append(items)

            line = file_data.readline()
            line_idx += 1
    # the last one
    if feature_items:
        if zip_format:
            yield list(zip(*feature_items))
        else:
            yield feature_items
    file_data.close()


def iob_iobes(tags):
    """IOB -> IOBES
    Args:
        tags: list(str)

    Returns:
        new_tags: list(str)
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def main():
    op = OptionParser()
    op.add_option('-i', '--input', dest='input', type='str', help='bio文件路径')
    op.add_option('-o', '--output', dest='output', type='str', help='bieo文件路径')
    argv = [] if not hasattr(sys.modules['__main__'], '__file__') else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if not opts.input or not opts.output:
        op.print_help()
        exit()
    path_input = opts.input
    path_output = opts.output
    file_output = codecs.open(path_output, 'w', encoding='utf-8')
    for i, items in enumerate(read_conllu(path_input)):
        items[-1] = iob_iobes(items[-1])
        items = list(zip(*items))
        for item in items:
            file_output.write('{0}\n'.format(' '.join(item)))
        file_output.write('\n')

        sys.stdout.write('sentence: {0}\r'.format(i))
        sys.stdout.flush()
    sys.stdout.write('sentence: {0}\r'.format(i+1))
    sys.stdout.flush()

    file_output.close()
    print('done!')


if __name__ == '__main__':
    main()
