import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only

import utils
import callbacks

class _logger:
    def __init__(self, save_path):
        self.save_path = save_path
    
    def log(self, obj, filename='log.txt', terminal_log=True, file_log=True):
        if terminal_log:
            print(obj)
        if file_log:
            with open(os.path.join(self.save_path, filename), 'a') as f:
                print(obj, file=f)

@callbacks.register('base-logger')
class BaseLogger(Callback):
    def __init__(self, save_path, config):
        super().__init__()

        self._timer = utils.Timer()
        self._start_epoch = None
        self.save_path = save_path
        self.config = config

    @property
    def state_key(self) -> str:
        return self._generate_state_key()
    
    @rank_zero_only
    def on_init_start(self, trainer):
        self._logger = _logger(utils.set_save_path(self.save_path))
        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)
    
    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        if self._start_epoch is None:
            self._start_epoch = trainer.current_epoch
        self._t_epoch_start = self._timer.t()
    
    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        t = self._timer.t()
        epoch_max = trainer.max_epochs

        prog = (trainer.current_epoch - self._start_epoch + 1) / (trainer.max_epochs - self._start_epoch + 1)
        t_epoch = utils.time_text(t - self._t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        met = trainer.progress_bar_callback.get_metrics(trainer, pl_module)
        met.pop('v_num')

        metric_str = ','.join([f'{k}={v}'for k, v in met.items()])
        self._logger.log(f'Epoch {trainer.current_epoch}/{trainer.max_epochs} {metric_str} {t_epoch} {t_elapsed}/{t_all}')