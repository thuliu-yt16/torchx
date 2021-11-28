import os
import json
from PIL import Image

import functools
import random
import math
import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl


import datasets
from datasets import register

from utils import to_pixel_samples

@register('sr-data-module')
class SRDataModule(pl.LightningDataModule):
    def __init__(self, train_spec, val_spec, batch_size):
        super().__init__()
        self.train_spec = train_spec
        self.val_spec = val_spec
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = datasets.make(self.train_spec)
            self.val_set = datasets.make(self.val_spec)

    def prepare_data(self):
        # download
        pass
    
    def general_loader(self, dataset, tag):
        return DataLoader(
            dataset, 
            shuffle=(tag=='train'), 
            num_workers=4,
            batch_size=self.batch_size if isinstance(self.batch_size, int) else self.batch_size[tag] ,
            pin_memory=True,
            )
    
    def train_dataloader(self):
        return self.general_loader(self.train_set, 'train')

    def val_dataloader(self):
        return self.general_loader(self.val_set, 'val')

    # def test_dataloader(self):
    #     return self.general_loader(mnist_wrapper(self.mnist_test), 'test')



@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = datasets.make(dataset_spec)
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset_spec, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = datasets.make(dataset_spec)
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.inp_size is None:
            s = random.uniform(self.scale_min, self.scale_max) # (1, 4)
        else:
            s_max = min((img.shape[-2] - 1) / self.inp_size, (img.shape[-1] - 1) / self.inp_size, self.scale_max)
            s = random.uniform(self.scale_min, s_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size # 48
            w_hr = round(w_lr * s) # (48, 192)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous()) # (-1, 1) coordinates in patch

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


# @register('sr-implicit-uniform-varied')
# class SRImplicitUniformVaried(Dataset):

#     def __init__(self, dataset, size_min, size_max=None,
#                  augment=False, gt_resize=None, sample_q=None):
#         self.dataset = dataset
#         self.size_min = size_min
#         if size_max is None:
#             size_max = size_min
#         self.size_max = size_max
#         self.augment = augment
#         self.gt_resize = gt_resize
#         self.sample_q = sample_q

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         img_lr, img_hr = self.dataset[idx]
#         p = idx / (len(self.dataset) - 1)
#         w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
#         img_hr = resize_fn(img_hr, w_hr)

#         if self.augment:
#             if random.random() < 0.5:
#                 img_lr = img_lr.flip(-1)
#                 img_hr = img_hr.flip(-1)

#         if self.gt_resize is not None:
#             img_hr = resize_fn(img_hr, self.gt_resize)

#         hr_coord, hr_rgb = to_pixel_samples(img_hr)

#         if self.sample_q is not None:
#             sample_lst = np.random.choice(
#                 len(hr_coord), self.sample_q, replace=False)
#             hr_coord = hr_coord[sample_lst]
#             hr_rgb = hr_rgb[sample_lst]

#         cell = torch.ones_like(hr_coord)
#         cell[:, 0] *= 2 / img_hr.shape[-2]
#         cell[:, 1] *= 2 / img_hr.shape[-1]

#         return {
#             'inp': img_lr,
#             'coord': hr_coord,
#             'cell': cell,
#             'gt': hr_rgb
#         }




@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
