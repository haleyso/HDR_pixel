import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import lpips
# import pytorch_colors as colors
import kornia
import sys


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, weightings, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, weightings, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        if 'lpips_loss' in config['loss']:
            self.lpips_loss_fn = lpips.LPIPS(net='vgg').to(self.device)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, sample in enumerate(self.data_loader):
            raw_patch = sample['raw_patch'].to(self.device)  
            jpg_patch = sample['jpg_patch'].to(self.device)  

            # torch.autograd.set_detect_anomaly(True)
            self.optimizer.zero_grad()
            output = self.model(raw_patch)

            loss = 0.0
            for crit in range(len(self.criterion)):
                if self.config['loss'][crit] == 'lpips_loss':
                    loss += self.weightings[crit] * self.criterion[crit](output, jpg_patch, self.lpips_loss_fn)
                else:
                    loss += self.weightings[crit] * self.criterion[crit](output, jpg_patch, self.device)
            # loss = loss/self.config['data_loader']['args']['batch_size']
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, jpg_patch))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                    
                self.writer.add_image('input', make_grid(raw_patch.cpu(), nrow=4, normalize=True))
                self.writer.add_image('guess', make_grid(output.cpu(), nrow=4, normalize=True))
                self.writer.add_image('actual', make_grid(jpg_patch.cpu(), nrow=4, normalize=True))
                    

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):

                raw_patch = sample['raw_patch'].to(self.device)  
                jpg_patch = sample['jpg_patch'].to(self.device) 
                raw_image = sample['raw_image'].to(self.device)
                jpg_image = sample['jpg_image'].to(self.device)
            
                output = self.model(raw_patch)
                output_image = self.model(raw_image)

                loss = 0.0
                for crit in range(len(self.criterion)):
                    if self.config['loss'][crit] == 'lpips_loss':
                        loss += self.weightings[crit] * self.criterion[crit](output, jpg_patch, self.lpips_loss_fn)
                    else:
                        loss += self.weightings[crit] * self.criterion[crit](output, jpg_patch, self.device)
                
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, jpg_patch))

                self.writer.add_image('input', make_grid(raw_patch.cpu(), nrow=4, normalize=True))
                self.writer.add_image('guess', make_grid(output.cpu(), nrow=4, normalize=True))
                self.writer.add_image('actual', make_grid(jpg_patch.cpu(), nrow=4, normalize=True))

                self.writer.add_image('Full Image Input', make_grid(raw_image.cpu(), nrow=4, normalize=True))
                self.writer.add_image('Full Image Guess', make_grid(output_image.cpu(), nrow=4, normalize=True))
                self.writer.add_image('Full Image Actual', make_grid(jpg_image.cpu(), nrow=4, normalize=True))



        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)



    