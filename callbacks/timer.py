import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_only

import utils
import callbacks

@callbacks.register('timer')
class Timer(Callback):
    def __init__(self, log):
        super().__init__()
        self._timer = utile.Timer()
        self._start_epoch = None
    
    @property
    def state_key(self) -> str:
        return self._generate_state_key()
    
    @rank_zero_only
    def on_epoch_start(self, trainer, pl_module):
        if self._start_epoch is None:
            self._start_epoch = trainer.current_epoch
        self._t_epoch_start = self._timer.t()
    
    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        t = self._timer.t()
        epoch_max = trainer.max_epochs

        prog = (trainer.current_epoch - self._start_epoch + 1) / (trainer.max_epochs - self._start_epoch + 1)
        t_epoch = utils.time_text(t - self._t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

        print(f'epoch {trainer.current_epoch}/{trainer.max_epochs} {t_epoch} {t_elapsed}/{t_all}')