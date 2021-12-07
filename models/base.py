import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import models
import losses
from models import register

import pytorch_lightning as pl


@models.register('base')
class BaseWrapper(pl.LightningModule):
    def __init__(self, model_spec, loss_spec, optim_spec, scheduler_spec=None):
        super().__init__()
        self.model = models.make(model_spec)
        self.loss = losses.make(loss_spec)

        self.optim_spec = optim_spec
        self.scheduler_spec = scheduler_spec

        self.save_hyperparameters()
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss(pred=out, batch=batch, batch_idx=batch_idx, stage='train', model=self.model)
        self.log('loss', loss['loss'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        OptimClass = getattr(torch.optim, self.optim_spec['name'])
        optim = OptimClass(self.model.parameters(), **self.optim_spec['args'])

        ret = {
            'optimizer': optim,
        }

        if self.scheduler_spec is not None:
            LRScheduler = getattr(torch.optim.lr_scheduler, self.scheduler_spec['name'])
            lr_scheduler = LRScheduler(optim, **self.scheduler_spec['args'])
            ret.update({
                'lr_scheduler': lr_scheduler,
            })

        return ret
    
    def training_step_end(self, training_step_outputs):
        return {'loss': training_step_outputs['loss'].mean()}

    def training_epoch_end(self, training_step_outputs):
        # for pred in training_step_outputs:
        #     ...
        pass
    
    def validation_step_end(self, batch_parts):
        # # predictions from each GPU
        # predictions = batch_parts["pred"]
        # # losses from each GPU
        # losses = batch_parts["loss"]

        # gpu_0_prediction = predictions[0]
        # gpu_1_prediction = predictions[1]

        # # do something with both outputs
        # return (losses[0] + losses[1]) / 2
        pass

    def validation_epoch_end(self, validation_step_outputs):
        # for out in validation_step_outputs:
        #     ...
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        # x, _ = batch

        # # encode
        # # for predictions, we could return the embedding or the reconstruction or both based on our need.
        # x = x.view(x.size(0), -1)
        # return self.encoder(x)
        pass

    # def test_step_end(...):
    #     pass

    # def test_epoch_end(...):
    #     pass

    # def any_extra_hook(...):
    #     pass

