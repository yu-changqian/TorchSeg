#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/9/28 下午12:13
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : init_func.py.py
import torch
import torch.nn as nn

# from seg_opr.dcn import DeformConvPack, ModulatedDeformConvPackv2, \
#     SEModulatedDeformConvPackv2, SubPixelModulatedDeformConvPackv2
from seg_opr.dcn import ModulatedDeformConv, DeformConv, FixWeightDeformConv


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (
                nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d,
                ModulatedDeformConv)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr, no_decay_lr=None):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, DeformConv):
            group_decay.append(m.weight)
        elif isinstance(m, (ModulatedDeformConv, FixWeightDeformConv)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, (
        nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    lr = lr if no_decay_lr is None else no_decay_lr
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def group_bn_weight(weight_group, module, norm_layer, lr, bn_lr=None):
    group_decay = []
    group_no_decay_bn = []
    group_no_decay_nonbn = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay_nonbn.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay_nonbn.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay_bn.append(m.weight)
            if m.bias is not None:
                group_no_decay_bn.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay_bn) + len(group_no_decay_nonbn)
    weight_group.append(dict(params=group_decay, lr=lr))
    # lr = lr if no_decay_lr is None else no_decay_lr
    weight_group.append(
        dict(params=group_no_decay_nonbn, weight_decay=.0, lr=lr))
    lr = lr if bn_lr is None else bn_lr
    weight_group.append(
        dict(params=group_no_decay_bn, weight_decay=.0, lr=lr))
    return weight_group
