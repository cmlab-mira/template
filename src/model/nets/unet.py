import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets.base_net import BaseNet


class UNet(BaseNet):
    """The implementation of U-Net with some modifications (ref: https://arxiv.org/abs/1505.04597).
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list): The list of the number of feature maps.
    """
    def __init__(self, in_channels, out_channels, num_features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features

        self.in_module = InModule(in_channels, num_features[0])
        self.down_module1 = DownModule(num_features[0], num_features[1])
        self.down_module2 = DownModule(num_features[1], num_features[2])
        self.down_module3 = DownModule(num_features[2], num_features[3])
        self.down_module4 = DownModule(num_features[3], num_features[4])
        self.up_module1 = UpModule(num_features[4], num_features[3])
        self.up_module2 = UpModule(num_features[3], num_features[2])
        self.up_module3 = UpModule(num_features[2], num_features[1])
        self.up_module4 = UpModule(num_features[1], num_features[0])
        self.out_module = OutModule(num_features[0], out_channels)

    def forward(self, input):
        features1 = self.in_module(input)
        features2 = self.down_module1(features1)
        features3 = self.down_module2(features2)
        features4 = self.down_module3(features3)
        features = self.down_module4(features4)

        features = self.up_module1(features, features4)
        features = self.up_module2(features, features3)
        features = self.up_module3(features, features2)
        features = self.up_module4(features, features1)
        output = self.out_module(features)
        return output


class InModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
    def forward(self, input):
        return self.conv(input)


class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))
    def forward(self, input):
        return self.conv(input)


class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True))

    def forward(self, input, features):
        input = self.deconv(input)

        h_diff = features.size(2) - input.size(2)
        w_diff = features.size(3) - input.size(3)
        input = F.pad(input, (w_diff // 2, w_diff - w_diff//2,
                              h_diff // 2, h_diff - h_diff//2))

        output = self.conv(torch.cat([input, features], dim=1))
        return output


class OutModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, input):
        return self.conv(input)
