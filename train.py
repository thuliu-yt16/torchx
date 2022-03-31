import argparse
import os
import sys
import copy
import random
import time
import shutil
from PIL import Image

import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import torchvision
from einops import repeat, rearrange, reduce, parse_shape
import datasets
import losses
import models
import utils
from models import make_lr_scheduler

import warnings
warnings.filterwarnings("ignore")

def seed_all(seed):
    log(f'Global seed set to {seed}')
    random.seed(seed) # Python
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu vars

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if type(v) is torch.Tensor:
            log('  {}: shape={}, dtype={}'.format(k, tuple(v.shape), v.dtype))
        else:
            log('  {}: type={}'.format(k, type(v)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=spec['shuffle'], num_workers=8, pin_memory=True)
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_training():
    if config.get('resume') is not None:
        log('resume from {}'.format(config['resume']))
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=config['load_optimizer'])
        
        lr_scheduler = make_lr_scheduler(optimizer, config.get('scheduler'))
        
        if config.get('run_step') is not None and config['run_step']:
            epoch_start = sv_file['epoch'] + 1
            for _ in range(epoch_start - 1):
                lr_scheduler.step()
        else:
            epoch_start = 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        lr_scheduler = make_lr_scheduler(optimizer, config.get('scheduler'))

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))

    loss_fn = losses.make(config['loss'])
    return model, optimizer, epoch_start, lr_scheduler, loss_fn

def train(train_loader, model, optimizer, loss_fn):
    model.train()
    train_loss = utils.Averager()

    with tqdm(train_loader, leave=False, desc='train') as pbar:
        for batch in pbar:
            for k, v in batch.items():
                if type(v) is torch.Tensor:
                    batch[k] = v.cuda()
            
            out = model(batch)

            list_of_loss = loss_fn(out, batch)
            loss = list_of_loss['loss']
            train_loss.add(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = None
            for k, v in list_of_loss.items():
                list_of_loss[k] = f'{v.item():.6f}'
            pbar.set_postfix({
                **list_of_loss,
            })

    return train_loss.item()


def val(val_loader, model):
    pass

def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)

    seed_all(config['seed'])
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    ckpt_path = os.path.join(save_path, 'ckpt')
    img_path = os.path.join(save_path, 'img')
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler, loss_fn = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')

    timer = utils.Timer()

    val(val_loader, model)

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, loss_fn)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        model_spec = copy.deepcopy(config['model'])
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = copy.deepcopy(config['optimizer'])
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
            'config': config,
        }

        torch.save(sv_file, os.path.join(ckpt_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(ckpt_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model
            
            val(val_loader, model_)

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    if 'seed' not in config:
        config['seed'] = int(time.time() * 1000) % 1000

    config['cmd_args'] = sys.argv
    config['resume'] = args.resume

    main(config, save_path)
