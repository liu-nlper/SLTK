#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import codecs
import yaml


file_yml = codecs.open('../configs/demo.train.yml')
configs = yaml.load(file_yml)

path = configs['data_params']['path_train']
print(os.path.abspath(path))

print(configs['data_params']['feature_names'])
print(configs['data_params']['path_pretrain'])

print(configs['model_params']['l2_rate'])
print(type(configs['model_params']['l2_rate']))
print(type(configs['model_params']['dropout_rate']))
