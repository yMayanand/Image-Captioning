import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from loss import CaptionLoss
from model import Encoder, Decoder
from preprocess import Tokenizer
from dataset import CaptionDataset
from torchvision import transforms
from torchtext.data.metrics import bleu_score 
from utils import cust_collate
from tqdm.notebook import tqdm
from argparse import ArgumentParser

import pytorch_lightning as pl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        transforms.Resize((400, 400)),
        transforms.ToTensor()
    ])

    tokenizer = Tokenizer(args.root_dir)
    tokenizer.tokenize(os.path.join(args.root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt'))

    encoder = Encoder().to(device)
    decoder = Decoder(tokenizer).to(device)

    train_ds = CaptionDataset(args.root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt', tokenizer, transform=tfms)
    val_ds = CaptionDataset(args.root_dir, 'Flicker8k_text/Flickr_8k.devImages.txt', tokenizer, transform=tfms)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=cust_collate)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, pin_memory=True, num_workers=2, collate_fn=cust_collate)

    criterion = CaptionLoss(decoder, 0.5, tokenizer)
    params = list(decoder.parameters()) #+ list(encoder.parameters()) + 
    #param_groups = [{'params': encoder.parameters(), 'lr': 1e-5},
                    #{'params': decoder.parameters(), 'lr': 1e-2}]

    optimizer = optim.Adam(params, lr=args.lr)
    #sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_dl), epochs=10)
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
            #sched.step()
            optimizer.zero_grad()
            temp_store.append(loss.item())
        loss_store.append(np.mean(temp_store))
        print(f"Epoch: {i} Loss: {loss_store[i]}")
        
    torch.save({'decoder_weights': decoder.state_dict()}, os.path.join(args.root_dir, 'model1.pt'))
    plt.plot(loss_store)
    plt.savefig(os.path.join(args.root_dir, 'loss.png'))

class Model(pl.LightningModule):
    def __init__(self, root_dir):
        super().__init__()
        tfms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])

        self.tokenizer = Tokenizer(root_dir)
        self.tokenizer.tokenize(os.path.join(root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt'))

        self.decoder = Decoder(self.tokenizer)
        self.register_buffer("encoder", Encoder())

        self.train_ds = CaptionDataset(root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt', self.tokenizer, transform=tfms)
        self.val_ds = CaptionDataset(root_dir, 'Flicker8k_text/Flickr_8k.devImages.txt', self.tokenizer, transform=tfms)

        self.loss_func = CaptionLoss(self.decoder, 0.5, self.tokenizer)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        out = self.encoder(x)
        loss = self.loss_func(out, y)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        out = self.encoder(x)
        loss = self.loss_func(out, y)

        score_list = []
        bs, dim = out.shape
        for  i in range(bs):
            candidate_corpus = [self.decode_one_sample(out[i]).split()]
            reference_corpus = [self.tokenizer.idx2val[i.item()] for i in y]
            reference_corpus = [[reference_corpus]]

            # get the bleu score
            score = bleu_score(candidate_corpus, reference_corpus)
            score_list.append(score)

            # average the score across all item
        avg_score = np.mean(score_list)

        self.log('val_bleu_score', avg_score)
        self.log('val_loss', loss)

    def decode_one_sample(self, im_hid):
        inp = self.decoder.emb(torch.LongTensor([self.tokenizer.val2idx['START']]).to(self.device))
        tot = 0
        seq = 0
        gen_caps = []
        self.decoder.init_states(im_hid[0])
        while True:
            out, *(self.decoder.hn, self.decoder.cn) = self.decoder(inp, self.decoder.hn, self.decoder.cn)
            _, idx = torch.max(out, dim=1)
            pred_token = self.tokenizer.idx2val[idx.item()]
            gen_caps.append(pred_token)
            tot += 1
            if (tot > 25) or (pred_token=='STOP'):
                break
            inp = self.decoder.emb(torch.LongTensor([self.tokenizer.val2idx[pred_token]]).to(self.device))
        return " ".join(gen_caps)


    def predict_step():
        pass


    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.val_ds, batch_size=32, pin_memory=True, num_workers=2, collate_fn=cust_collate)
        return loader

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(self.train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=cust_collate)
        return loader


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", default="./", type=str, help="root directory path")
    parser.add_argument("--epochs", default=10, type=int, help="num of epochs to train")

    args = parser.parse_args()
    model = Model(args.root_dir)
    trainer = pl.Trainer(accelerator='gpu', gpus=1, max_epochs=args.epochs)
    trainer.fit(model)

#load_from_ckpt(encoder, decoder, './checkpoint/caption.pt')

# TODO: move this function to another module
def predict(encoder, decoder,  tokenizer, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = encoder.to(device)
    encoder.eval()

    decoder = decoder.to(device)
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

def evaluate(root_dir, ds_path, model_path):
    tfms = transforms.Compose([
        transforms.Resize((400, 400)),
        transforms.ToTensor()
    ])

    tokenizer = Tokenizer(root_dir)
    tokenizer.tokenize(os.path.join(root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt'))

    # encoder and decoder
    encoder = Encoder()
    decoder = Decoder(tokenizer)

    state_dict = torch.load(model_path)
    decoder.load_state_dict(state_dict['decoder_weights'])

    # create dataset
    ds = CaptionDataset(root_dir, ds_path, tokenizer, transform=tfms)

    score_list = []

    # get a data item
    for i in tqdm(range(len(ds))):
        # get an item from dataset
        data, label = ds[i]

        # unsqueeze data for fake batch dimension
        data = data.unsqueeze(0)

        # predict on that item
        candidate_corpus = [predict(encoder, decoder,  tokenizer, data).split()]

        reference_corpus = [tokenizer.idx2val[i.item()] for i in label]
        reference_corpus = [[reference_corpus]]

        # get the bleu score
        score = bleu_score(candidate_corpus, reference_corpus)
        score_list.append(score)

    # average the score across all item
    avg_score = np.mean(score_list)
    print(f"bleu score is {avg_score}")
    return avg_score
