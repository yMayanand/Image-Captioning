import GPUtil
import torch
from pytorch_lightning.callbacks import Callback
from torch.nn.utils.rnn import pad_sequence


def get_gpu_usage():
    gpu = GPUtil.getGPUs()[0]
    gpu_load = gpu.load * 100
    gpu_memory_util = gpu.memoryUtil * 100
    return gpu_load, gpu_memory_util


def train_collate(batch):
    xs, ys, zs = list(zip(*batch))
    xs = torch.stack(xs)
    ys = pad_sequence(ys, batch_first=True, padding_value=3)
    zs = torch.stack(zs)
    return xs, ys, zs


def val_collate(batch):
    xs, ys, zs = list(zip(*batch))
    xs = torch.stack(xs)
    return xs, ys, zs

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomMetricLogger(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val = pl_module.val_bleu_meter.avg
        trainer.logger.log_metrics({'val_bleu_score': val}, trainer.global_step)
        pl_module.val_bleu_meter.reset()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        if trainer.global_step % 10 == 0:
            for name, param in pl_module.model.named_parameters():
                if param.requires_grad:
                    trainer.logger.experiment.add_histogram(name, param.grad, trainer.global_step)


# TODO: dataset --> dataloader --> forward --> backward
class TrainingStats(Callback):
    def __init__(self):
        pass

    def setup(self, trainer, pl_module, stage):
        pass

    def teardown(self, trainer, pl_module, stage):
        pass

    def on_fit_start(self, trainer, pl_module):
        pass

    def on_fit_end(self, trainer, pl_module):
        pass

    def on_sanity_check_start(self, trainer, pl_module):
        pass

    def on_sanity_check_end(self, trainer, pl_module):
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass

    def on_train_start(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        pass

    def on_validation_start(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        pass

    def on_after_backward(self, trainer, pl_module):
        pass

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        pass

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pass
