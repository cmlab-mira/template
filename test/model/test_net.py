import pytest
import torch
import torch.nn as nn
import numpy as np
from box import Box
from pathlib import Path
from src.model.nets.base_net import BaseNet


class MyNet(BaseNet):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(self.in_channels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, self.out_channels, kernel_size=3)


class TestNetClass():
    @classmethod
    def setup_class(self):
        self.cfg = Box.from_yaml(filename=Path("test/configs/test_config.yaml"))

    def test_base_net(self):
        net = BaseNet()

    def test_my_net(self):
        net = MyNet(**self.cfg.net.kwargs)