from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from seg_opr.seg_oprs import ConvBnRelu, SeparableConvBnRelu, SELayer
from utils.pyt_utils import load_model

__all__ = ['XSNet', 'iwxsnet18', 'iwxsnet1x34']


class ContextEmbedding(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(ContextEmbedding, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, in_channels, 1, 1, 0,
                                         has_bn=True, norm_layer=norm_layer,
                                         has_relu=False, has_bias=False)

    def forward(self, x):
        shortcut = x
        x = self.pooling(x)
        global_context = x
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        x = F.interpolate(x, size=shortcut.size()[2:],
                          mode='bilinear', align_corners=True)

        return x + shortcut, global_context


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride,
                 dilation=1, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.use_res_connect = (stride == 1)
        self.in_channels = in_channels

        if stride == 1:
            in_channels = in_channels // 2
            mid_channels = round(in_channels * expansion)
            out_channels = in_channels
        elif stride == 2:
            mid_channels = out_channels
        self.residual_branch = nn.Sequential(
            ConvBnRelu(in_channels, mid_channels, 3, 1, 1, dilation,
                       has_relu=True, norm_layer=norm_layer),
            SeparableConvBnRelu(mid_channels, out_channels, 3, stride,
                                dilation, dilation,
                                has_relu=False, norm_layer=norm_layer))

    def forward(self, x):
        if self.use_res_connect:
            shortcut, x = torch.split(x, self.in_channels // 2, dim=1)
            return torch.cat([self.residual_branch(x), shortcut], dim=1)
        else:
            return self.residual_branch(x)


class XSNet(nn.Module):
    def __init__(self, block, layers, channels, norm_layer=nn.BatchNorm2d,
                 context_embedding=ContextEmbedding, return_global=False):
        super(XSNet, self).__init__()

        self.return_global = return_global
        self.in_channels = 24
        self.conv1 = ConvBnRelu(3, self.in_channels, 3, 2, 1,
                                has_bn=True, norm_layer=norm_layer,
                                has_relu=False, has_bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, norm_layer,
                                       layers[0], channels[0], 4, stride=2)
        self.layer2 = self._make_layer(block, norm_layer,
                                       layers[1], channels[1], 4, stride=2)
        self.layer3 = self._make_layer(block, norm_layer,
                                       layers[2], channels[2], 4, stride=2)
        self.context_embedding = context_embedding(channels[2], norm_layer)

    def _make_layer(self, block, norm_layer, blocks,
                    out_channels, expansion, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, expansion,
                            stride=stride, norm_layer=norm_layer))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, expansion,
                                stride=1, norm_layer=norm_layer))
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x);
        blocks.append(x)
        x = self.layer2(x);
        blocks.append(x)
        x = self.layer3(x);
        blocks.append(x)
        x, global_context = self.context_embedding(x);
        blocks.append(x)

        if self.return_global:
            return blocks, global_context
        return blocks


def xsnet16(pretrained_model=None, **kwargs):
    model = XSNet(Block, [2, 2, 2], [32, 64, 96], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def iwxsnet18(pretrained_model=None, **kwargs):
    model = XSNet(Block, [2, 2, 4], [32, 64, 96], **kwargs)
    # model = XSNet(Block, [2, 4, 4], [32, 64, 96], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def xsnet1x18(pretrained_model=None, **kwargs):
    model = XSNet(Block, [2, 2, 4], [32 * 2, 64 * 2, 96 * 2], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def xsnet18_v1b(pretrained_model=None, **kwargs):
    model = XSNet(Block, [2, 3, 3], [32, 64, 96], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def xsnet34(pretrained_model=None, **kwargs):
    model = XSNet(Block, [4, 4, 8], [32, 64, 96], **kwargs)
    # model = XSNet(Block, [2, 4, 4], [32, 64, 96], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def iwxsnet1x34(pretrained_model=None, **kwargs):
    model = XSNet(Block, [4, 4, 8], [32 * 2, 64 * 2, 96 * 2], **kwargs)
    # model = XSNet(Block, [2, 4, 4], [32, 64, 96], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def xsnet22(pretrained_model=None, **kwargs):
    model = XSNet(Block, [2, 2, 6], [32, 64, 96], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def xsnet23(pretrained_model=None, **kwargs):
    model = XSNet(Block, [3, 4, 4], [32, 64, 96], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def xsnet1x23(pretrained_model=None, **kwargs):
    model = XSNet(Block, [3, 4, 4], [32 * 2, 64 * 2, 96 * 2], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model
