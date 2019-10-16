# encoding: utf-8
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from base_model import resnet50
from seg_opr.seg_oprs import ConvBnRelu


class CPNet(nn.Module):
    def __init__(self, out_planes, criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(CPNet, self).__init__()
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                 bn_eps=config.bn_eps,
                                 bn_momentum=config.bn_momentum,
                                 deep_stem=True, stem_width=64)
        self.backbone.layer3.apply(partial(self._nostride_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self._nostride_dilate, dilate=4))

        self.business_layer = []

        self.context = ObjectContext(2048, 512, norm_layer)

        self.head_layer = nn.Sequential(
            ConvBnRelu(2048 + 2048, 512, 3, 1, 1,
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
        self.business_layer.append(self.context)
        self.business_layer.append(self.head_layer)
        self.business_layer.append(self.aux_layer)

        self.criterion = criterion
        self.bce_criterion = nn.BCELoss(reduction='mean')

    def forward(self, data, label=None, aux_label0=None, aux_label1=None):
        blocks = self.backbone(data)

        fm, large_map, small_map = self.context(blocks[-1])

        fm = self.head_layer(fm)
        fm = F.interpolate(fm, scale_factor=8, mode='bilinear',
                           align_corners=True)
        softmax_fm = F.log_softmax(fm, dim=1)

        aux_fm = self.aux_layer(blocks[-2])
        aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)

        if label is not None:
            main_loss = self.criterion(fm, label)
            aux_loss = self.criterion(aux_fm, label)
            sim_loss0 = self.bce_criterion(large_map, aux_label0)
            sim_loss1 = self.bce_criterion(small_map, aux_label1)
            loss = main_loss + 0.4 * aux_loss + sim_loss0 + sim_loss1
            return loss

        return softmax_fm

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


class ObjectContext(nn.Module):
    def __init__(self, in_channels, inner_channel, norm_layer=nn.BatchNorm2d):
        super(ObjectContext, self).__init__()
        self.in_channels = in_channels
        self.inner_channel = inner_channel

        self.reduce_conv = ConvBnRelu(self.in_channels, self.inner_channel,
                                      1, 1, 0,
                                      has_bn=True, has_relu=True,
                                      has_bias=False, norm_layer=norm_layer)

        self.ocb0 = ObjectContextBranch(self.inner_channel, 60, norm_layer)
        self.ocb1 = ObjectContextBranch(self.inner_channel, 20, norm_layer)

    def forward(self, x):
        value = self.reduce_conv(x)

        large_fm, large_sim_map = self.ocb0(value)
        small_fm, small_sim_map = self.ocb1(value)

        output = torch.cat([x, large_fm, small_fm], dim=1)
        return output, large_sim_map, small_sim_map


class ObjectContextBranch(nn.Module):
    def __init__(self, in_channels, size, norm_layer=nn.BatchNorm2d):
        super(ObjectContextBranch, self).__init__()
        self.in_channels = in_channels
        # self.inner_channel = inner_channel
        self.size = size
        self.point_num = size ** 2

        self.pool = nn.AdaptiveAvgPool2d(self.size)
        self.intra_similarity_branch = nn.Sequential(
            ConvBnRelu(self.in_channels, self.in_channels, 1, 1, 0,
                       has_bn=True, has_relu=True,
                       has_bias=False, norm_layer=norm_layer),
            ConvBnRelu(self.in_channels, self.point_num, 1, 1, 0,
                       has_bn=True, has_relu=False,
                       has_bias=False, norm_layer=norm_layer),
        )

        self.intra_post_conv = ConvBnRelu(self.in_channels,
                                          self.in_channels,
                                          1, 1, 0, has_bn=True, has_relu=True,
                                          has_bias=False, norm_layer=norm_layer)
        self.inter_post_conv = ConvBnRelu(self.in_channels,
                                          self.in_channels,
                                          1, 1, 0, has_bn=True, has_relu=True,
                                          has_bias=False, norm_layer=norm_layer)

    def forward(self, x):
        value = self.pool(x)
        b, h, w = value.size(0), value.size(2), value.size(3)

        intra_similarity_map = self.intra_similarity_branch(value)
        intra_similarity_map = intra_similarity_map.view(b, h * w, -1)
        intra_similarity_map = intra_similarity_map.permute(0, 2, 1)
        intra_similarity_map = torch.sigmoid(intra_similarity_map)

        inter_similarity_map = 1 - intra_similarity_map

        value = value.view(b, self.in_channels, -1)
        value = value.permute(0, 2, 1)

        intra_context = torch.bmm(intra_similarity_map, value)
        intra_context = intra_context.div(self.point_num)
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(b, self.in_channels, h, w)
        intra_context = self.intra_post_conv(intra_context)

        inter_context = torch.bmm(inter_similarity_map, value)
        inter_context = inter_context.div(self.point_num)
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(b, self.in_channels, h, w)
        inter_context = self.inter_post_conv(inter_context)

        output = torch.cat([intra_context, inter_context], dim=1)

        if h < x.size(2):
            output = F.interpolate(output,
                                   size=(x.size(2), x.size(3)),
                                   mode='bilinear', align_corners=True)

        return output, intra_similarity_map


if __name__ == "__main__":
    model = PSPNet(150, None)
    print(model)
