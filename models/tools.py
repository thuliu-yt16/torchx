import copy


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

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def make_lr_scheduler(optim, scheduler_spec):
    if scheduler_spec is None:
        return None
    auto_create = [
        'CosineAnnealingLR',
        'ExponentialLR',
        # 'LinearLR',
        'MultiStepLR',
        'ReduceLROnPlateau',
        'StepLR',
    ]

    name = scheduler_spec['name']
    if name in auto_create:
        LRCLASS = getattr(torch.optim.lr_scheduler, name)
        lr_scheduler = LRCLASS(optim, **scheduler_spec['args'])
        return lr_scheduler
    elif name == 'LambdaLR':
        args = copy.deepcopy(scheduler_spec['args'])
        lr_lambda = make_lambda(args['lr_lambda'])
        args['lr_lambda'] = lr_lambda
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, **args)
        return lr_scheduler
    else:
        raise NotImplementedError

def warmup_lr(start, end=1.0, total_iters=5, exp_gamma=1):
    def _warmup(epoch):
        if epoch + 1 >= total_iters:
            return end * exp_gamma ** (epoch + 1 - total_iters)
        else:
            return epoch / (total_iters - 1) * (end - start) + start 

    return _warmup

def make_lambda(lr_lambda):
    FUNC = {
        'warmup_lr': warmup_lr,
    }[lr_lambda['name']]
    return FUNC(**lr_lambda['args'])



