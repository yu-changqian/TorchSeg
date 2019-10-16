import numpy as np
import scipy.ndimage as nd

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.logger import get_logger
from seg_opr.seg_oprs import one_hot

logger = get_logger()


class SigmoidFocalLoss(nn.Module):
    def __init__(self, ignore_label, gamma=2.0, alpha=0.25,
                 reduction='mean'):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        b, h, w = target.size()
        pred = pred.view(b, -1, 1)
        pred_sigmoid = pred.sigmoid()
        target = target.view(b, -1).float()
        mask = (target.ne(self.ignore_label)).float()
        target = mask * target
        onehot = target.view(b, -1, 1)

        # TODO: use the pred instead of pred_sigmoid
        max_val = (-pred_sigmoid).clamp(min=0)

        pos_part = (1 - pred_sigmoid) ** self.gamma * (
                pred_sigmoid - pred_sigmoid * onehot)
        neg_part = pred_sigmoid ** self.gamma * (max_val + (
                (-max_val).exp() + (-pred_sigmoid - max_val).exp()).log())

        loss = -(self.alpha * pos_part + (1 - self.alpha) * neg_part).sum(
            dim=-1) * mask
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [1.4297, 1.4805, 1.4363, 3.365, 2.6635, 1.4311, 2.1943, 1.4817,
                 1.4513, 2.1984, 1.5295, 1.6892, 3.2224, 1.4727, 7.5978, 9.4117,
                 15.2588, 5.6818, 2.2067])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = torch.sort(mask_prob)
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class PiecewiseProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False, epoch_thresh=None):
        super(PiecewiseProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        self.epoch_thresh = epoch_thresh
        if use_weight:
            weight = torch.FloatTensor(
                [1.4297, 1.4805, 1.4363, 3.365, 2.6635, 1.4311, 2.1943, 1.4817,
                 1.4513, 2.1984, 1.5295, 1.6892, 3.2224, 1.4727, 7.5978, 9.4117,
                 15.2588, 5.6818, 2.2067])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target, num_epoch):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        if num_epoch > self.epoch_thresh:
            prob = F.softmax(pred, dim=1)
            prob = (prob.transpose(0, 1)).reshape(c, -1)

            if self.min_kept > num_valid:
                logger.info('Labels: {}'.format(num_valid))
            elif num_valid > 0:
                prob = prob.masked_fill_(1 - valid_mask, 1)
                mask_prob = prob[
                    target, torch.arange(len(target), dtype=torch.long)]
                threshold = self.thresh
                if self.min_kept > 0:
                    index = mask_prob.argsort()
                    threshold_index = index[min(len(index), self.min_kept) - 1]
                    if mask_prob[threshold_index] > self.thresh:
                        threshold = mask_prob[threshold_index]
                    kept_mask = mask_prob.le(threshold)
                    target = target * kept_mask.long()
                    valid_mask = valid_mask * kept_mask
                    # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

            target = target.masked_fill_(1 - valid_mask, self.ignore_label)

        target = target.view(b, h, w)
        return self.criterion(pred, target)


class FocalProb(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 gamma=2):
        super(FocalProb, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.gamma = gamma

        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                   ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            logger.info('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(1 - valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class AutoOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', drop_ratio=0.3):
        super(AutoOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.drop_ratio = float(drop_ratio)

        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                   ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()

        prob = F.softmax(pred, dim=1)
        prob = prob.view(b, c, -1)
        similarity = torch.matmul(prob.permute(0, 2, 1), prob)
        similarity = torch.sum(similarity, dim=2) / (h * w)
        sorted_similarity, _ = torch.sort(similarity, dim=1, descending=True)
        prob_threshold = sorted_similarity[:,
                         int(h * w * self.drop_ratio)].view(b, 1)
        kept_mask = similarity.lt(prob_threshold).view(-1)
        valid_mask = valid_mask * kept_mask
        target = target.masked_fill_(1 - valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class PriorLoss(nn.Module):
    def __init__(self, scale, num_class, ignore_index):
        super(PriorLoss, self).__init__()
        self.scale = scale
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.criterion = torch.nn.BCELoss(reduction='none')

    def forward(self, pred, target):
        b, h, w = target.size()
        scaled_gts = F.interpolate((target.view(b, 1, h, w)).float(),
                                   scale_factor=self.scale,
                                   mode="nearest")

        valid_mask = torch.ones_like(scaled_gts)
        valid_mask[scaled_gts == self.ignore_index] = 0
        valid_vector = valid_mask.view(b, -1, 1)
        valid_mask = torch.bmm(valid_vector, valid_vector.permute(0, 2, 1))

        scaled_gts[scaled_gts == self.ignore_index] = self.num_class
        scaled_gts = scaled_gts.squeeze_()
        C = self.num_class + 1
        one_hot_gts = one_hot(scaled_gts, C).view(b, C, -1)
        similarity_gts = torch.bmm(one_hot_gts.permute(0, 2, 1),
                                   one_hot_gts)

        bce_loss = self.criterion(pred, similarity_gts)
        num_valid = valid_mask.sum()
        num_valid = torch.where(num_valid > 0, num_valid,
                                torch.ones(1, device=num_valid.device))
        bce_loss = valid_mask * bce_loss
        bce_loss = bce_loss.sum() / num_valid

        valid_vector = valid_vector.view(b, -1)
        num_valid = valid_vector.sum()
        num_valid = torch.where(num_valid > 0, num_valid,
                                torch.ones(1, device=num_valid.device))

        vtarget = similarity_gts * valid_mask

        precision_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(pred, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = self.criterion(precision_part, precision_label)
        precision_loss = valid_vector * precision_loss
        precision_loss = precision_loss.sum() / num_valid

        recall_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = self.criterion(recall_part, recall_label)
        recall_loss = valid_vector * recall_loss
        recall_loss = recall_loss.sum() / num_valid

        vtarget = (1 - similarity_gts) * valid_mask
        spec_part = torch.sum((1 - pred) * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = self.criterion(spec_part, spec_label)
        spec_loss = valid_vector * spec_loss
        spec_loss = spec_loss.sum() / num_valid

        loss = bce_loss + recall_loss + spec_loss + precision_loss

        return loss


class MaskBCELoss(nn.Module):
    def __init__(self, mask, reduction='mean'):
        super(MaskBCELoss, self).__init__()
        self.mask = mask
        self.criterion = torch.nn.BCELoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred, target):
        original_loss = self.criterion(pred, target)
        self.mask = self.mask.to(original_loss.get_device())
        num_valid = self.mask.sum()
        loss = self.mask * original_loss

        if self.reduction == 'mean':
            loss = loss.sum() / num_valid

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        iflat = input.view(-1)
        tflat = target.view(-1)
        iou = (iflat * tflat).sum()
        negtive_iou = ((1 - iflat) * (1 - tflat)).sum()

        score = 1 - ((2. * iou + self.smooth) /
                     (iflat.sum() + tflat.sum() + self.smooth)) - (
                        (negtive_iou + self.smooth) / (
                        2 - iflat.sum() - tflat.sum() + self.smooth))
        score /= input.size(0)

        return score


class DiceLossv2(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLossv2, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        iflat = input.view(-1)
        tflat = target.view(-1)
        iou = (iflat * tflat).sum()
        negtive_iou = ((1 - iflat) * (1 - tflat)).sum()

        score = 1 - ((3 * iou * negtive_iou + self.smooth) / (
                (negtive_iou * (iflat.sum() + tflat.sum())) + iou * (
            (1 - iflat).sum()) + self.smooth))
        #
        # score = 1 - ((2. * iou + self.smooth) /
        #              (iflat.sum() + tflat.sum() + self.smooth)) - (
        #                 (negtive_iou + self.smooth) / (
        #                 2 - iflat.sum() - tflat.sum() + self.smooth))
        score /= input.size(0)

        return score


class AntimagnetLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(AntimagnetLoss, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        diagonal_matrix = (1 - torch.eye(target.size(1))).to(
            target.get_device())
        vtarget = diagonal_matrix * target

        attract_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        attract_part = attract_part.div_(denominator)
        attract_label = torch.ones_like(attract_part)
        attract_loss = self.criterion(attract_part, attract_label)

        repel_part = torch.sum((1 - pred) * (1 - target), dim=2)
        denominator = torch.sum(1 - target, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        repel_part = repel_part.div_(denominator)
        repel_label = torch.ones_like(repel_part)
        repel_loss = self.criterion(repel_part, repel_label)

        loss = attract_loss + repel_loss

        return loss


class AntimagnetLossv2(nn.Module):
    def __init__(self, reduction='mean'):
        super(AntimagnetLossv2, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        diagonal_matrix = (1 - torch.eye(target.size(1))).to(
            target.get_device())
        vtarget = diagonal_matrix * target

        attract_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        attract_part = attract_part.div(denominator)
        attract_label = torch.ones_like(attract_part)
        attract_loss = self.criterion(attract_part, attract_label)

        repel_part = torch.sum((1 - pred) * (1 - target), dim=2)
        denominator = torch.sum(1 - target, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        repel_part = repel_part.div(denominator)
        repel_label = torch.ones_like(repel_part)
        repel_loss = self.criterion(repel_part, repel_label)

        # print(attract_part, repel_part)
        interact_part = 1 - torch.abs(attract_part - repel_part)
        interact_label = torch.ones_like(interact_part)
        interact_loss = self.criterion(interact_part, interact_label)

        loss = attract_loss + repel_loss + interact_loss

        return loss


class AntimagnetLossv3(nn.Module):
    def __init__(self, reduction='mean'):
        super(AntimagnetLossv3, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        diagonal_matrix = (1 - torch.eye(target.size(1))).to(
            target.get_device())
        vtarget = diagonal_matrix * target

        attract_part = pred * vtarget
        base_count = (torch.sum(vtarget, dim=2, keepdim=True) * 0.3).long()
        base_prob, _ = torch.sort(attract_part, dim=2, descending=True)
        base_prob = base_prob.gather(dim=2, index=base_count)
        attract_mask = torch.le(attract_part, base_prob).float() * vtarget
        attract_part *= attract_mask
        attract_part = torch.sum(attract_part, dim=2)
        denominator = torch.sum(attract_mask, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        attract_part = attract_part.div(denominator)
        attract_label = torch.ones_like(attract_part)
        attract_loss = self.criterion(attract_part, attract_label)

        repel_part = (1 - pred) * (1 - target)
        base_count = (torch.sum(1 - target, dim=2, keepdim=True) * 0.3).long()
        base_prob, _ = torch.sort(repel_part, dim=2, descending=True)
        base_prob = base_prob.gather(dim=2, index=base_count)
        repel_mask = torch.le(repel_part, base_prob).float() * (1 - target)
        repel_part *= repel_mask
        repel_part = torch.sum(repel_part, dim=2)
        denominator = torch.sum(repel_mask, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        repel_part = repel_part.div(denominator)
        repel_label = torch.ones_like(repel_part)
        repel_loss = self.criterion(repel_part, repel_label)

        loss = attract_loss + repel_loss

        return loss


class AntimagnetLossv4(nn.Module):
    def __init__(self, reduction='mean'):
        super(AntimagnetLossv4, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        diagonal_matrix = (1 - torch.eye(target.size(1))).to(
            target.get_device())
        vtarget = diagonal_matrix * target

        attract_part = torch.sum(pred * vtarget)
        denominator = torch.sum(vtarget)
        denominator = 1 if denominator == 0 else denominator
        attract_part = attract_part.div(denominator)
        attract_label = torch.ones_like(attract_part)
        attract_loss = self.criterion(attract_part, attract_label)

        repel_part = torch.sum((1 - pred) * (1 - target))
        denominator = torch.sum(1 - target)
        denominator = 1 if denominator == 0 else denominator
        repel_part = repel_part.div(denominator)
        repel_label = torch.ones_like(repel_part)
        repel_loss = self.criterion(repel_part, repel_label)

        loss = attract_loss + repel_loss

        return loss


class AntimagnetLossv5(nn.Module):
    def __init__(self, reduction='mean'):
        super(AntimagnetLossv5, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        diagonal_matrix = (1 - torch.eye(target.size(1))).to(
            target.get_device())
        vtarget = diagonal_matrix * target

        attract_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        attract_part = attract_part.div(denominator)
        attract_label = torch.ones_like(attract_part)
        attract_loss = self.criterion(attract_part, attract_label)

        repel_part = (1 - pred) * (1 - target)
        repel_part = torch.max(repel_part - 0.5, torch.zeros_like(repel_part))
        repel_part = torch.sum(repel_part, dim=2)
        denominator = torch.sum(1 - target, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        repel_part = repel_part.div(denominator)
        repel_label = torch.ones_like(repel_part)
        repel_loss = self.criterion(repel_part, repel_label)

        loss = attract_loss + repel_loss

        return loss


class AntimagnetLossv6(nn.Module):
    def __init__(self, reduction='mean'):
        super(AntimagnetLossv6, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        diagonal_matrix = (1 - torch.eye(target.size(1))).to(
            target.get_device())
        vtarget = diagonal_matrix * target

        recall_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = self.criterion(recall_part, recall_label)

        spec_part = torch.sum((1 - pred) * (1 - target), dim=2)
        denominator = torch.sum(1 - target, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = self.criterion(spec_part, spec_label)

        precision_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(pred, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = self.criterion(precision_part, precision_label)

        loss = recall_loss + spec_loss + precision_loss

        return loss


class AntimagnetLossv7(nn.Module):
    def __init__(self, reduction='mean'):
        super(AntimagnetLossv7, self).__init__()
        self.reduction = reduction
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, pred, target):
        bce_loss = self.criterion(pred, target)

        diagonal_matrix = (1 - torch.eye(target.size(1))).to(
            target.get_device())
        vtarget = diagonal_matrix * target

        recall_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = self.criterion(recall_part, recall_label)

        spec_part = torch.sum((1 - pred) * (1 - target), dim=2)
        denominator = torch.sum(1 - target, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = self.criterion(spec_part, spec_label)

        precision_part = torch.sum(pred * vtarget, dim=2)
        denominator = torch.sum(pred, dim=2)
        denominator = denominator.masked_fill_(1 - (denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = self.criterion(precision_part, precision_label)

        loss = bce_loss + recall_loss + spec_loss + precision_loss

        return loss
