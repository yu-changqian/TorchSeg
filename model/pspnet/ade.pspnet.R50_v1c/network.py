# encoding: utf-8
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from base_model import resnet50
from seg_opr.seg_oprs import ConvBnRelu


class PSPNet(nn.Module):
    def __init__(self, out_planes, criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(PSPNet, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64)
        self.backbone.layer3.apply(partial(self._nostride_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self._nostride_dilate, dilate=4))

        self.business_layer = []
        self.psp_layer = PyramidPooling('psp', out_planes, 2048,
                                        norm_layer=norm_layer)
        self.aux_layer = nn.Sequential(
            ConvBnRelu(1024, 1024, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(1024, out_planes, kernel_size=1)
        )
        self.business_layer.append(self.psp_layer)
        self.business_layer.append(self.aux_layer)

        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)

        psp_fm = self.psp_layer(blocks[-1])
        aux_fm = self.aux_layer(blocks[-2])

        psp_fm = F.interpolate(psp_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)
        aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)
        psp_fm = F.log_softmax(psp_fm, dim=1)
        aux_fm = F.log_softmax(aux_fm, dim=1)

        if label is not None:
            loss = self.criterion(psp_fm, label)
            aux_loss = self.criterion(aux_fm, label)
            loss = loss + 0.4 * aux_loss
            return loss

        return psp_fm

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
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


class PyramidPooling(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 ConvBnRelu(fc_dim, 512, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False,
                            norm_layer=norm_layer))
            ])))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()
        ppm_out = [x]
        for pooling in self.ppm:
            ppm_out.append(
                F.interpolate(pooling(x), size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True))
        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out


if __name__ == "__main__":
    model = PSPNet(150, None)
    print(model)
