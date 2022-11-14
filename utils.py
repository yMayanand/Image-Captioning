import GPUtil
import torch
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.callbacks import Callback


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


# TODO: dataset --> dataloader --> forward --> backward
class TrainingStats(Callback):
    def __init__(self) -> None:
        super().__init__()