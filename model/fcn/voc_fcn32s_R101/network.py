# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import config
from base_model import resnet101
from seg_opr.seg_oprs import ConvBnRelu


class FCN(nn.Module):
    def __init__(self, out_planes, criterion, inplace=True,
                 pretrained_model=None, norm_layer=nn.BatchNorm2d):
        super(FCN, self).__init__()
        self.backbone = resnet101(pretrained_model, inplace=inplace,
                                  norm_layer=norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)

        self.business_layer = []
        self.head = _FCNHead(2048, out_planes, inplace, norm_layer=norm_layer)
        self.aux_head = _FCNHead(1024, out_planes, inplace,
                                 norm_layer=norm_layer)

        self.business_layer.append(self.head)
        self.business_layer.append(self.aux_head)

        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)
        fm = self.head(blocks[-1])
        pred = F.interpolate(fm, scale_factor=32, mode='bilinear',
                             align_corners=True)

        aux_fm = self.aux_head(blocks[-2])
        aux_pred = F.interpolate(aux_fm, scale_factor=16, mode='bilinear',
                                 align_corners=True)

        if label is not None:
            loss = self.criterion(pred, label)
            aux_loss = self.criterion(aux_pred, label)
            loss = loss + config.aux_loss_ratio * aux_loss
            return loss

        return pred


class _FCNHead(nn.Module):
    def __init__(self, in_planes, out_planes, inplace=True,
                 norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_planes = in_planes // 4
        self.cbr = ConvBnRelu(in_planes, inter_planes, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)
        self.dropout = nn.Dropout2d(0.1)
        self.conv1x1 = nn.Conv2d(inter_planes, out_planes, kernel_size=1,
                                 stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.conv1x1(x)
        return x


if __name__ == "__main__":
    model = FCN(21, None)
    print(model)
