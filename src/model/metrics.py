import torch
import torch.nn as nn


class Dice(nn.Module):
    ''' Dice score
    Args:
        num_classes (int): the number of the prediction class
    '''

    def __init__(self, num_classes):
        self.__name__ = 'Dice'
        self.num_classes = num_classes

    def forward(self, pred, label):
        # One hot the prediction and the ground truth label
        _pred = torch.zeros((pred.shape[0], self.num_classes, pred.shape[2:])).scatter_(1, pred, 1).to(pred.device)
        _label = torch.zeros_like(_pred).scatter_(1, label, 1)

        # Calculate the dice score
        reduce_dim = pred.shape[2:]
        intersection = 2.0 * (_pred * _label).sum(reduce_dim)
        union = (_pred).sum(reduce_dim) + (_label).sum(reduce_dim)
        epsilon = 1e-10
        score = intersection / (union + epsilon)

        return torch.mean(score, dim=0)
