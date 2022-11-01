import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from loss import CaptionLoss
from model import Encoder, Decoder
from preprocess import tokenizer
from dataset import CaptionDataset
from torchvision import transforms
from utils import cust_collate
from tqdm import tqdm
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder().to(device)
decoder = Decoder().to(device)

def load_from_ckpt(encoder, decoder, path):
    weights = torch.load(path, map_location=device)
    encoder.load_state_dict(weights['encoder'])
    decoder.load_state_dict(weights['decoder'])

def train_model():
    parser = ArgumentParser()

    parser.add_argument("--root_dir", default="./", type=str, help="root directory path")
    parser.add_argument("--epochs", default=10, type=int, help="num of epochs to train")
    parser.add_argument("--lr", default=1e-2, type=float, help="learing rate")
    
    args = parser.parse_args()

    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_ds = CaptionDataset(os.path.join(args.root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt'), transform=tfms)
    val_ds = CaptionDataset(os.path.join(args.root_dir, 'Flicker8k_text/Flickr_8k.devImages.txt'), transform=tfms)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=cust_collate)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, pin_memory=True, num_workers=2, collate_fn=cust_collate)

    criterion = CaptionLoss(decoder, 0.5)
    params = list(decoder.parameters()) #+ list(encoder.parameters()) + 
    optimizer = torch.optim.Adam(params, lr=args.lr)
    sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_dl), epochs=10)
    epochs = args.epochs
    loss_store = []

    for i in tqdm(range(epochs)):
        temp_store = []
        for data, labels in tqdm(train_dl, total=len(train_dl), leave=False):
            encoder.train()
            decoder.train()
            data = data.to(device)
            labels = labels
            im_hid = encoder(data)
            loss = criterion(im_hid, labels)
            loss.backward()
            optimizer.step()
            sched.step()
            optimizer.zero_grad()
            temp_store.append(loss.item())
        loss_store.append(np.mean(temp_store))
        print(f"Epoch: {i} Loss: {loss_store[i]}")
        
    torch.save({'decoder_weights': decoder.state_dict()}, os.path.join(args.root_dir, 'model1.pt'))
    plt.plot(loss_store)
    plt.savefig(os.path.join(args.root_dir, 'loss.png'))

if __name__ == '__main__':
    train_model()

#load_from_ckpt(encoder, decoder, './checkpoint/caption.pt')

# TODO: move this function to another module
def predict(data):
    encoder.eval()
    decoder.eval()
    data = data.to(device)
    im_hid = encoder(data)
    inp = decoder.emb(torch.LongTensor([tokenizer.val2idx['START']]).to(device))
    tot = 0
    seq = 0
    gen_caps = []
    decoder.init_states(im_hid[0])
    while True:
        out, *(decoder.hn, decoder.cn) = decoder(inp, decoder.hn, decoder.cn)
        _, idx = torch.max(out, dim=1)
        pred_token = tokenizer.idx2val[idx.item()]
        gen_caps.append(pred_token)
        tot += 1
        if (tot > 25) or (pred_token=='STOP'):
            break
        inp = decoder.emb(torch.LongTensor([tokenizer.val2idx[pred_token]]).to(device))
    return " ".join(gen_caps)