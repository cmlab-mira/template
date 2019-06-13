import torch

from src.runner.trainers.base_trainer import BaseTrainer


class KitsClfTrainer(BaseTrainer):
    """The KiTS trainer for classification task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_iter(self, batch):
        """Run an iteration to obtain the output and the losses.
        Args:
            batch (dict or tuple): A batch of data.
        Returns:
            output (torch.Tensor): The computed output.
            losses (sequence of torch.Tensor): The computed losses.
        """
        image, label = batch['image'], batch['label']
        image = torch.cat([image, image, image], dim=1) # Concatenate three one-channel images to a three-channels image.
        output = self.net(image)
        losses = tuple(loss(output, label) for loss in self.losses)
        return output, losses
