import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
import datasets

class mnist_wrapper(Dataset):
    def __init__(self, mnist):
        self._dataset = mnist
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, idx):
        img, label = self._dataset[idx]
        return {
            'img': img,
            'label': label,
        }


@datasets.register('mnist')
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = torchvision.datasets.MNIST(self.root_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = torchvision.datasets.MNIST(self.root_dir, train=False, transform=self.transform)

    def prepare_data(self):
        # download
        torchvision.datasets.MNIST(self.root_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.root_dir, train=False, download=True)
    
    def general_loader(self, dataset, tag):
        return DataLoader(
            dataset, 
            shuffle=(tag=='train'), 
            num_workers=os.cpu_count(), 
            batch_size=self.batch_size,
            pin_memory=True,
            )
    
    def train_dataloader(self):
        return self.general_loader(mnist_wrapper(self.mnist_train), 'train')

    def val_dataloader(self):
        return self.general_loader(mnist_wrapper(self.mnist_val), 'val')

    def test_dataloader(self):
        return self.general_loader(mnist_wrapper(self.mnist_test), 'test')




