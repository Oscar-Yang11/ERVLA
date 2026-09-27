import gc

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class GCCallback(Callback):
    """Periodic garbage collection at key training milestones.

    Disables Python's automatic GC on training start and manually collects
    only at controlled milestones, preventing unpredictable GC pauses during
    forward/backward. Always pairs gc.collect() with torch.cuda.empty_cache().

    Args:
        gc_interval: Number of training steps between full GC sweeps.
    """

    def __init__(self, gc_interval: int = 1000):
        super().__init__()
        self.gc_interval = gc_interval

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        gc.disable()

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        gc.enable()

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx):
        # Only GC at the end of an optimizer step, not every micro-batch.
        if (batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return
        step = trainer.global_step
        if step > 0 and step % self.gc_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        gc.collect()
        torch.cuda.empty_cache()

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint):
        gc.collect()
        torch.cuda.empty_cache()
