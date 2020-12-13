import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1

    def forward(self, input, target):
        flat_input = input.contiguous().view(-1)
        flat_target = target.contiguous().view(-1)
        numerator = 2 * torch.dot(flat_input, flat_target) + self.smooth
        denominator = torch.pow(flat_input, 2).sum() + torch.pow(flat_target, 2).sum() + self.smooth
        return 1 - (numerator / denominator)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss.mean()


class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        return loss
