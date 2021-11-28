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

# import warnings
# warnings.filterwarnings("ignore")

# def make_data_loader(spec, tag=''):
#     if spec is None:
#         return None

#     dataset = datasets.make(spec['dataset'])
#     dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

#     log('{} dataset: size={}'.format(tag, len(dataset)))
#     for k, v in dataset[0].items():
#         log('  {}: shape={}'.format(k, tuple(v.shape)))

#     loader = DataLoader(dataset, batch_size=spec['batch_size'],
#         shuffle=(tag == 'train'), num_workers=os.cpu_count(), pin_memory=True)
#     return loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
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
    log, writer = utils.set_save_path(save_path)

    if 'seed' not in config:
        config['seed'] = int(time.time() * 1000) % 1000
    config['cmd_args'] = vars(args)

    # save config
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    pl.seed_everything(config['seed'])
    n_gpus = len(args.gpu.split(','))

    data_module = datasets.make(config['data_module'])
    model = models.make(config['train_wrapper'])
    trainer = pl.Trainer(
        gpus=n_gpus,
        strategy=pl.plugins.DDPPlugin(find_unused_parameters=False),
        **config['trainer_params']
        )
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()