import torch

from src.runner.trainers.base_trainer import BaseTrainer


class KitsClfTrainer(BaseTrainer):
    """The KiTS trainer for classification task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict or tuple): A batch of data.

        Returns:
            input (torch.Tensor): The data input.
            target (torch.Tensor): The data target.
        """
        image, label = batch['image'], batch['label']
        image = torch.cat([image, image, image], dim=1) # Concatenate three one-channel images to a three-channels image.
        return image, label

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.Tensor): The data target.

        Returns:
            losses (sequence of torch.Tensor): The computed losses.
        """
        losses = [loss(output, target) for loss in self.losses]
        return losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.Tensor): The data target.

        Returns:
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        metrics = [metric(output, target) for metric in self.metrics]
        return metrics
