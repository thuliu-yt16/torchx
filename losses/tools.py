import copy

losses = {}

def register(name):
    def decorator(cls):
        losses[name] = cls
        return cls
    return decorator


def make(loss_spec, args=None):
    loss_args = copy.deepcopy(loss_spec.get('args', {}))
    args = args or {}
    loss_args.update(args)

    loss = losses[loss_spec['name']](**loss_args)
    return loss