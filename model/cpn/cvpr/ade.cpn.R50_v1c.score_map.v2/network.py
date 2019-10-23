# encoding: utf-8
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_model import resnet50
from config import config
from seg_opr.loss_opr import AntimagnetLossv6
from seg_opr.seg_oprs import ConvBnRelu


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

        self.context = ObjectContext(2048, 512, out_planes, norm_layer)

        self.head_layer = nn.Sequential(
            ConvBnRelu(2048 + 1024, out_planes, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(out_planes, out_planes, kernel_size=1)
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

        fm, local_fm, intra_sim_map = self.context(blocks[-1])

        fm = self.head_layer(fm)
        fm = F.interpolate(fm, scale_factor=8, mode='bilinear',
                           align_corners=True)
        softmax_fm = F.softmax(fm, dim=1)

        aux_fm = self.aux_layer(blocks[-2])

        if label is not None:
            aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear',
                                   align_corners=True)
            local_fm = F.interpolate(local_fm, scale_factor=8, mode='bilinear',
                                     align_corners=True)

            main_loss = self.criterion(fm, label)
            aux_loss = self.criterion(aux_fm, label)
            aux_loss2 = self.criterion(local_fm, label)

            intra_sim_loss = self.bce_criterion(intra_sim_map, aux_label)

            loss = main_loss + 0.4 * aux_loss + aux_loss2 + intra_sim_loss
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
        self.t1 = nn.Conv2d(in_channels, in_channels, kernel_size=(ksize, 1),
                            stride=1, padding=(padding, 0))
        self.t2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, ksize),
                            stride=1, padding=(0, padding))
        self.p1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, ksize),
                            stride=1, padding=(0, padding))
        self.p2 = nn.Conv2d(in_channels, in_channels, kernel_size=(ksize, 1),
                            stride=1, padding=(padding, 0))
        self.bn = norm_layer(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p2(x2)

        output = self.relu(self.bn(x1 + x2))
        return output


class ObjectContext(nn.Module):
    def __init__(self, in_channels, inner_channels, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(ObjectContext, self).__init__()
        self.in_channels = in_channels
        self.inner_channels = inner_channels
        self.out_planes = out_planes

        self.local_aggregation = nn.Sequential(
            ConvBnRelu(self.in_channels, self.inner_channels,
                       3, 1, 1,
                       has_bn=True, has_relu=True,
                       has_bias=False, norm_layer=norm_layer),
            SymmetricConv(self.inner_channels, 11, norm_layer)
        )

        self.score_proj = nn.Conv2d(self.inner_channels, self.out_planes,
                                    kernel_size=1, stride=1, padding=0)

        self.intra_post_conv = ConvBnRelu(self.inner_channels,
                                          self.inner_channels,
                                          1, 1, 0, has_bn=True, has_relu=True,
                                          has_bias=False, norm_layer=norm_layer)
        self.inter_post_conv = ConvBnRelu(self.inner_channels,
                                          self.inner_channels,
                                          1, 1, 0, has_bn=True, has_relu=True,
                                          has_bias=False, norm_layer=norm_layer)

    def forward(self, x):
        b, h, w = x.size(0), x.size(2), x.size(3)

        local_x = self.local_aggregation(x)
        softmax_x = F.softmax(self.score_proj(local_x), dim=1)

        value = softmax_x.view(b, self.out_planes, -1)
        similarity_map = torch.bmm(value.permute(0, 2, 1), value)
        print(similarity_map.max(), similarity_map.min(), similarity_map.mean())
        similarity_map = (self.out_planes**-.5) * similarity_map
        # similarity_map = torch.sigmoid(similarity_map)

        inter_similarity_map = 1 - similarity_map

        local_x = (local_x.view(b, self.inner_channels, -1)).permute(0, 2, 1)

        intra_context = torch.bmm(similarity_map, local_x)
        intra_context = intra_context.div(3600)
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(b, self.inner_channels, *x.size()[2:])
        intra_context = self.intra_post_conv(intra_context)

        inter_context = torch.bmm(inter_similarity_map, local_x)
        inter_context = inter_context.div(3600)
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(b, self.inner_channels, *x.size()[2:])
        inter_context = self.inter_post_conv(inter_context)

        output = torch.cat([x, intra_context, inter_context], dim=1)
        return output, softmax_x, similarity_map


if __name__ == "__main__":
    model = PSPNet(150, None)
    print(model)