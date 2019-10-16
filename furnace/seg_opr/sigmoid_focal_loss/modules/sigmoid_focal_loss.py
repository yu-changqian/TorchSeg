from torch import nn

from ..functions.sigmoid_focal_loss import sigmoid_focal_loss


class SigmoidFocalLoss(nn.Module):

    def __init__(self, ignore_label, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        assert logits.is_cuda
        b, h, w = targets.size()
        logits = logits.view(b, -1)
        targets = targets.view(b, -1)
        mask = (targets.ne(self.ignore_label))
        targets = mask.long() * targets
        target_mask = (targets > 0)

        loss = sigmoid_focal_loss(logits, targets, self.gamma, self.alpha,
                                  'none')
        loss = loss * mask.float()
        return loss.sum() / target_mask.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
