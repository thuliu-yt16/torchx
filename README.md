# torchx

Torchx is a general framework for deep learning experiments under PyTorch based on pytorch-lightning. 

This project has taken inspiration from [LIIF](https://github.com/yinboc/liif).

## Feature

- [x] custom dataset/loss/model/training wrapper
- [x] config/code auto copy
- [x] checkpoint load/save
- [x] ETA for training
- [ ] local file logger
- [ ] custom callback
- [ ] gan-like training wrapper

## Get Started

### Requirements

- python 3.6+
- torch, torchvision, pytorch-lightning
- tensorboardX, PyYAML

### Typical directory tree

```sh
torchx
├── callbacks 
│   ├── base_logger.py              # ETA for training, the experiment directory creator
│   ├── __init__.py                                            
│   ├── progressbar.py
│   ├── ...
│   └── tools.py                    # callback register
├── configs                                                            
│   ├── ...                         # your experiment configurations
│   └── mnist-classification.yaml                
├── datasets                                                        
│   ├── __init__.py
│   ├── mnist.py
│   ├── ...
│   └── tools.py                    # dataset register
├── load                            # datasets
│   ├── div2k 
│   ├── ...
│   └── mnist
├── losses
│   ├── __init__.py
│   ├── tools.py                    # loss register
│   ├── ...
│   └── myloss.py
├── models
│   ├── base.py                     # definition of base training wrapper
│   ├── __init__.py
│   ├── mlp.py
│   ├── resnet.py
│   ├── ...
│   └── tools.py                    # model/training wrapper register
├── save                                                                
│   ├── ...
│   └── test                        # experiment directory
│       ├── ckpt                    # all the checkpoints
│       ├── config.yaml             # configuration of the experiment
│       ├── default                 # some other files
│       │   └── version_0
│       │       ├── events...       # tensorboard logger
│       │       └── hparams.yaml    # hyperparameter of training wrapper
│       └── src                     # code at the beginning of the experiment
├── train_pl.py                     # experiment entrance with pytorch lightning
├── train.py                        # experiment entrance
└── utils.py                        # utility functions
```

### Make a custom model

It is very easy to define a new model in torchx. All you need is to register the new model class with a name and create it with the  `make` function in a training wrapper. Check the decorator `register` and function `make` in `models/tools.py`:

```python
models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator
  
def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
```

To register a model class `MyModel`  defined in `models/mymodel.py`

1. add the decorator before the class definition:

```python
import models
@models.register('mymodel_name')
class MyModel(nn.Module):
  ...
```

2.  import `models/mymodel.py` in `models/__init__.py`:

```python
from . import mymodel
```

Similarly for losses/datasets/callbacks/training wrapper.

### Pytorch-lightning (pl) in torchx

The main reason to create torchx based on pl is that the distributed data parallel is well written in pl.

We use the `pl.LightningModule`as the training wrapper. For base training wrapper in `models/base.py`. It could recieve different models, losses, optimizers, learning-rate schedulers and launch the general training step. To make a  custom training wrapper with additional validation step or logs, just inherit `BaseWrapper` or implement a new one and register it. 

We implement some features with callbacks in pl, such as `BaseLogger` in `callbacks/base_logger.py`. It serves as an experiment directory creator and an ETA timer/logger for training.

We use `pl.Trainer` and `pl.callbacks.ModelCheckpoint` to manage the trainnig/validation procedure and model checkpointing, respectively. In the configuration, we specify their arguments under the field `trainer_params` and `checkpoint`. 

PL automatically moves tensor to proper devices so do not call `.cuda()` or `.to(device)` in your code. Instead, to create new tensor in your `nn.Module`, use `new_tensor.type_as(exist_tensor)` .

Some useful documents for your custom objects:

Callback API: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.base.html#module-pytorch_lightning.callbacks.base

Trainer Usage: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#

Trainer API: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html?highlight=Trainer#trainer

LightningModule API: https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#module-pytorch_lightning.core.lightning

Checkpoint: https://pytorch-lightning.readthedocs.io/en/stable/common/weights_loading.html?highlight=Checkpoint#checkpoint-saving

Hooks: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=on_train_epoch_start#hooks

### Configuration structure

Take `configs/mnist-classification.yaml` as an example:

```yaml
data_module:                        # define datasets (required)
  name: mnist                       # specify the dataset name 
  args:
    root_dir: ./load/mnist
    batch_size: 32
  
train_wrapper:                      # define training wrapper (required)
  name: base                        # specify the training wrapper name 
  args:                             # arguments in BaseWrapper.__init__()
    model_spec:                     # define model
      name: resnet18
      args:
        num_classes: 10
    
    loss_spec:                      # define loss
      name: mnist_ce                
    
    optim_spec:                     # define optimizer
      name: Adam
      args:
        lr: 1.e-4
        betas: [0.9, 0.999]

trainer_params:                     # arguments for pl.Trainer
  max_epochs: 100
  strategy: ddp                     # training strategy (dp, ddp)

checkpoint:                         # arguments for pl.callbacks.ModelCheckpoint
  every_n_epochs: 1                                 

seed: 42                            # fix seeds
```

### Logging

Logging in torchx is divided into three main parts:

- Metric log in `pl.LightningModule`: metrics like `loss`, `accuracy` could be logged with `pl.LightningModule.log/pl.LightningModule.log_dict` .Metrics are important for validation and model comparison. PL could trace metrics to select the best model in checkpoint saving routine.  
- Custom loggers in `pl.LightningModule `: pl integrates several loggers, such as [tensorboard](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html#module-pytorch_lightning.loggers.tensorboard), [csv](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.csv_logs.html#module-pytorch_lightning.loggers.csv_logs). These loggers are specified in `logger`  argument when `pl.Trainer` is created. They will log the metrics above automatically. You can also save images with the particular logger.(https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=logger#logger)
- Terminal logger as `Callback`: torchx implements a `BaseLogger` to print metrics/training ETA after epoch ends. It will also create a file `log.txt`  under the experiment directory. 

### Options for high frequency use

`trainer_params.strategy` : dp/ddp/ddp_spawn/ddp2. ddp_spawn is not recommend in pl. When `ddp` is used, ensure that `train.py` is directed executed from terminal.  See https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html for detail or instruction.
`trainer_params.log_every_n_steps`: how often to log within steps (defaults to every 50 steps).
`checkpoint.every_n_epochs`: how often to run checkpoint saving routine.
`checkpoint.monitor`: the metric to monitor to get the best model. 
`checkpoint.mode`: min/max. 


### Run a simple experiment

```bash
python train_pl.py --config configs/mnist-pl.yaml --gpu 1,2 --name test
```

Or you could train without pytorch_lightning and ddp using 

```bash
python train.py --config configs/mnist.yaml --gpu 1 --name test
```

