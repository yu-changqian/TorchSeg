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

        self.context = ObjectContext(2048, 512, norm_layer)

        self.head_layer = nn.Sequential(
            ConvBnRelu(2048 + 1024, 512, 3, 1, 1,
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
        self.antimagnet_criterion = AntimagnetLossv6()

    def forward(self, data, label=None, aux_label=None):
        blocks = self.backbone(data)

        fm, intra_sim_map = self.context(blocks[-1])

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
            intra_sim_loss = self.bce_criterion(intra_sim_map, aux_label)
            antimagnet_loss = self.antimagnet_criterion(intra_sim_map,
                                                        aux_label)
            loss = main_loss + 0.4 * aux_loss + intra_sim_loss + antimagnet_loss
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


class SymmetricConv(nn.Module):
    def __init__(self, in_channels, ksize, norm_layer=nn.BatchNorm2d):
        super(SymmetricConv, self).__init__()
        padding = ksize // 2
        self.t1 = nn.Conv2d(in_channels, in_channels, groups=in_channels,
                            kernel_size=(ksize, 1),
                            stride=1, padding=(padding, 0))
        self.t2 = nn.Conv2d(in_channels, in_channels, groups=in_channels,
                            kernel_size=(1, ksize),
                            stride=1, padding=(0, padding))
        self.p1 = nn.Conv2d(in_channels, in_channels, groups=in_channels,
                            kernel_size=(1, ksize),
                            stride=1, padding=(0, padding))
        self.p2 = nn.Conv2d(in_channels, in_channels, groups=in_channels,
                            kernel_size=(ksize, 1),
                            stride=1, padding=(padding, 0))
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, in_channels, 1, 1, 0,
                                         has_bn=True, norm_layer=norm_layer,
                                         has_relu=False, has_bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        residual = self.point_wise_cbr(self.bn(x1 + x2))
        output = self.relu(x + residual)
        return output


class ObjectContext(nn.Module):
    def __init__(self, in_channels, inner_channel, norm_layer=nn.BatchNorm2d):
        super(ObjectContext, self).__init__()
        self.in_channels = in_channels
        self.inner_channel = inner_channel

        self.reduce_conv = ConvBnRelu(self.in_channels, self.inner_channel,
                                      3, 1, 1,
                                      has_bn=True, has_relu=True,
                                      has_bias=False, norm_layer=norm_layer)

        self.intra_similarity_branch = nn.Sequential(
            SymmetricConv(self.inner_channel, 7, norm_layer),
            nn.Conv2d(self.inner_channel, 3600, 1, 1, 0, groups=16, bias=False),
            norm_layer(3600)
        )

        self.intra_post_conv = ConvBnRelu(self.inner_channel,
                                          self.inner_channel,
                                          1, 1, 0, has_bn=True, has_relu=True,
                                          has_bias=False, norm_layer=norm_layer)
        self.inter_post_conv = ConvBnRelu(self.inner_channel,
                                          self.inner_channel,
                                          1, 1, 0, has_bn=True, has_relu=True,
                                          has_bias=False, norm_layer=norm_layer)

    def forward(self, x):
        b, h, w = x.size(0), x.size(2), x.size(3)

        value = self.reduce_conv(x)

        intra_similarity_map = self.intra_similarity_branch(value)
        intra_similarity_map = intra_similarity_map.view(b, h * w, -1)
        intra_similarity_map = intra_similarity_map.permute(0, 2, 1)
        intra_similarity_map = torch.sigmoid(intra_similarity_map)

        inter_similarity_map = 1 - intra_similarity_map

        value = value.view(b, self.inner_channel, -1)
        value = value.permute(0, 2, 1)

        intra_context = torch.bmm(intra_similarity_map, value)
        intra_context = intra_context.div(3600)
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(b, self.inner_channel, *x.size()[2:])
        intra_context = self.intra_post_conv(intra_context)

        inter_context = torch.bmm(inter_similarity_map, value)
        inter_context = inter_context.div(3600)
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(b, self.inner_channel, *x.size()[2:])
        inter_context = self.inter_post_conv(inter_context)

        output = torch.cat([x, intra_context, inter_context], dim=1)
        return output, intra_similarity_map


if __name__ == "__main__":
    model = PSPNet(150, None)
    print(model)
