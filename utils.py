import torch
from torch.nn.utils.rnn import pad_sequence

def decode_caption(idx):
    temp = []
    for i in idx:
        temp1 = tokenizer.idx2val[i]
        temp.append(temp1)
    return ' '.join(temp)

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