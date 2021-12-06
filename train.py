import argparse
import os
import time

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

import datasets
import models
import utils

import pytorch_lightning as pl
from callbacks import BaseLogger, GlobalProgressBar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--resume')
    args = parser.parse_args()

    global config, log, writer

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name or '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    # create save directory and copy code
    # log, writer = utils.set_save_path(save_path)

    if 'seed' not in config:
        config['seed'] = int(time.time() * 1000) % 1000
    config['cmd_args'] = vars(args)

    # save config
    # with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
    #     yaml.dump(config, f, sort_keys=False)
    
    pl.seed_everything(config['seed'])
    n_gpus = len(args.gpu.split(','))

    data_module = datasets.make(config['data_module'])
    model = models.make(config['train_wrapper'])
    checkpoint_cfg = config.get('checkpoint', {})

    base_logger = BaseLogger(save_path, config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(save_path, 'ckpt'),
        save_last=True,
        auto_insert_metric_name=True,
        **checkpoint_cfg,
        )

    trainer = pl.Trainer(
        gpus=n_gpus,
        callbacks=[base_logger, checkpoint_callback],
        logger=False,
        **config['trainer_params']
        )
    
    if args.resume:
        trainer.fit(model, data_module, ckpt_path=args.resume)
    else:
        trainer.fit(model, data_module)

if __name__ == '__main__':
    main()