import torch

def decode_caption(idx):
    temp = []
    for i in idx:
        temp1 = tokenizer.idx2val[i]
        temp.append(temp1)
    return ' '.join(temp)

def cust_collate(batch):
    xs, ys = list(zip(*batch))
    xs = torch.stack(xs)
    return xs, ys