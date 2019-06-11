import torch
import torch.nn as nn
from tqdm import tqdm

from src.runner.trainers.base_trainer import BaseTrainer


class KitsSegTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').
        Returns:
            log (dict): The log information.
            batch (dict or tuple): The last batch of the data.
            output (torch.Tensor): The corresponding output.
        """
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()
        count = 0
        for batch in trange:
            if isinstance(batch, dict):
                batch = dict((key, data.to(self.device)) for key, data in batch.items())
            else:
                batch = tuple(data.to(self.device) for data in batch)

            if mode == 'training':
                output, losses = self._run_iter(batch)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    output, losses = self._run_iter(batch)
                    loss = (torch.stack(losses) * self.loss_weights).sum()

            if self.lr_scheduler is None:
                pass
            elif isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and mode == 'validation':
                self.lr_scheduler.step(loss)
            else:
                self.lr_scheduler.step()

            batch_size = output.size(0)
            log['Loss'] += loss.item() * batch_size
            for loss, _loss in zip(self.losses, losses):
                log[loss.__class__.__name__] += _loss.item() * batch_size
            for metric in self.metrics:
                scores = metric(output.argmax(dim=1, keepdim=True), batch['label'])
                for i, score in enumerate(scores):
                    log[f'{metric.__class__.__name__}_{i}'] += score.item() * batch_size
            count += batch_size
            trange.set_postfix(**dict((key, value / count) for key, value in log.items()))

        for key in log:
            log[key] /= count
        return log, batch, output

    def _run_iter(self, batch):
        image, label = batch['image'], batch['label']
        pred = self.net(image)
        losses = tuple(loss(pred, label) for loss in self.losses)

        return pred, losses

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
            for i in range(self.net.out_channels):
                log[f'{metric.__class__.__name__}_{i}'] = 0
        return log
