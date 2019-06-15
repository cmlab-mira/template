import torch
import torch.nn as nn
from tqdm import tqdm

from src.runner.trainers.base_trainer import BaseTrainer


class KitsSegTrainer(BaseTrainer):
    """The KiTS trainer for segmentation task.
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
        return batch['image'], batch['label']

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

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss in self.losses:
            log[loss.__class__.__name__] = 0
        for metric in self.metrics:
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] = 0
                for i in range(self.net.out_channels):
                    log[f'Dice_{i}'] = 0
            else:
                log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (sequence of torch.Tensor): The computed losses.
            metrics (sequence of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.losses, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metrics, metrics):
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] += _metric.mean().item() * batch_size
                for i, class_score in enumerate(_metric):
                    log[f'Dice_{i}'] += class_score.item() * batch_size
            else:
                log[metric.__class__.__name__] += _metric.item() * batch_size
