#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/4 下午11:32
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : gluon2pytorch.py

import os
import mxnet as mx
from gluoncv.model_zoo import get_model
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str)
parser.add_argument('-p', '--model_path', default='/unsullied/sharefs/yuchangqian/Storage/model_zoo', type=str)

args = parser.parse_args()
gluon_model_path = os.path.join(args.model_path, 'gluon_model')
gluon_model_file = os.path.join(gluon_model_path, args.model+'.params')
if not os.path.exists(gluon_model_file):
    gluon_model = get_model(args.model, pretrained=True, root=gluon_model_path)
    gluon_model_files = os.listdir(gluon_model_path)
    for file in gluon_model_files:
        if '-' in file:
            new_name = file.split('-')[0] + '.params'
            os.rename(os.path.join(gluon_model_path, file), os.path.join(gluon_model_path, new_name))
gluon_model_params = mx.nd.load(gluon_model_file)

pytorch_model_params = {}

print('Convert Gluon Model to PyTorch Model ...')
for key, value in gluon_model_params.items():
    if 'gamma' in key:
        key = key.replace('gamma', 'weight')
    elif 'beta' in key:
        key = key.replace('beta', 'bias')

    tensor = torch.from_numpy(value.asnumpy())
    tensor.require_grad = True
    pytorch_model_params[key] = tensor

pytorch_model_path = os.path.join(args.model_path, 'pytorch_model')
torch.save(pytorch_model_params, os.path.join(pytorch_model_path, args.model+'.pth'))
print('Finished!')
