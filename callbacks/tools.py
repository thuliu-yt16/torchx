import copy

callbacks = {}

def register(name):
    def decorator(cls):
        callbacks[name] = cls
        return cls
    return decorator

def make(cb_spec, args=None):
    cb_args = copy.deepcopy(cb_spec.get('args', {}))
    args = args or {}
    cb_args.update(args)

    cb = callbacks[cb_spec['name']](**cb_args)
    return cb
