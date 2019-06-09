import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    '''Dice loss
    '''
    def __init__(self):
        self.__name__ = 'DiceLoss'

    def forward(self, pred, label):
        # One hot the ground truth label
        # 2D image: N, num_classes, H, W, 3D volume: N, num_classes, H, W, D
        _label = torch.zeros_like(pred).scatter_(1, label, 1)

        # Calculate the dice loss
        reduce_dim = pred.shape[2:]
        intersection = 2.0 * (pred * _label).sum(reduce_dim)
        union = (pred ** 2).sum(reduce_dim) + (_label ** 2).sum(reduce_dim)
        epsilon = 1e-10
        score = intersection / (union + epsilon)

        return 1 - torch.mean(score)
