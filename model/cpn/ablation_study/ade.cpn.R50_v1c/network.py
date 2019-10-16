# encoding: utf-8
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from base_model import resnet50
from seg_opr.seg_oprs import ConvBnRelu
from seg_opr.loss_opr import AntimagnetLossv6


class CPNet(nn.Module):
    def __init__(self, out_planes, criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(CPNet, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64)
        self.generate_dilation(self.backbone.layer3, dilation=2)
        self.generate_dilation(self.backbone.layer4, dilation=4,
                               multi_grid=[1, 2, 4])

        self.business_layer = []

        self.head_layer = nn.Sequential(
            ConvBnRelu(2048, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )
        self.aux_layer = nn.Sequential(
            ConvBnRelu(1024, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )
        self.business_layer.append(self.head_layer)
        self.business_layer.append(self.aux_layer)

        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)

        fm = self.head_layer(blocks[-1])
        fm = F.interpolate(fm, scale_factor=8, mode='bilinear',
                           align_corners=True)
        softmax_fm = F.log_softmax(fm, dim=1)

        aux_fm = self.aux_layer(blocks[-2])
        aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)

        if label is not None:
            main_loss = self.criterion(fm, label)
            aux_loss = self.criterion(aux_fm, label)
            loss = main_loss + 0.4 * aux_loss
            return loss

        return softmax_fm

    def generate_dilation(self, module, dilation, multi_grid=None):
        for idx, block in enumerate(module):
            if multi_grid is None:
                grid = 1
            else:
                grid = multi_grid[idx % len(multi_grid)]
            dilation = dilation * grid
            block.apply(partial(self._nostride_dilate, dilate=dilation))

    @staticmethod
    def _nostride_dilate(m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == "__main__":
    model = PSPNet(150, None)
    print(model)
