import torch
import torch.nn as nn
import torchvision

from src.model.nets.base_net import BaseNet


class ClfNet(BaseNet):
    """The modified class for training from the scratch or finetuning the image classifiers as provided in torchvision.models.
    Args:
        num_classes (int): The number of output classes.
        name (str): The image classifiers name as provided in torchvision.models (ref: https://pytorch.org/docs/stable/torchvision/models.html).
        pretrained (bool): Whether to load the pretrained weights (default: True).
    """
    def __init__(self, num_classes, name, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.name = name
        self.pretrained = pretrained

        cls = getattr(torchvision.models, name)
        if pretrained:
            model = cls(pretrained=pretrained)

            # Get the attribute named classifier or fc in the model.
            classifier = getattr(model, 'classifier', None)
            fc = getattr(model, 'fc', None)

            # Change the output classes and reset the parameters in classifier or fc.
            if classifier:
                classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
                for module in classifier[:-1]:
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()
            elif fc:
                fc = nn.Linear(fc.in_features, num_classes)
        else:
            model = cls(pretrained=pretrained, num_classes=num_classes)
        setattr(self, name, model)

    def forward(self, input):
        return getattr(self, self.name)(input)
