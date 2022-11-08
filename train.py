import os
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from loss import CaptionLoss
from model import Encoder, Decoder, CaptionModel
from preprocess import Tokenizer
from dataset import CaptionDataset
from torchvision import transforms
from torchtext.data.metrics import bleu_score 
from utils import cust_collate
from tqdm.notebook import tqdm
from argparse import ArgumentParser

import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, root_dir):
        super().__init__()
        tfms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])

        self.tokenizer = Tokenizer(root_dir)
        self.tokenizer.tokenize(os.path.join(root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt'))

        self.model = CaptionModel(self.tokenizer)

        self.train_ds = CaptionDataset(root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt', self.tokenizer, transform=tfms)
        self.val_ds = CaptionDataset(root_dir, 'Flicker8k_text/Flickr_8k.devImages.txt', self.tokenizer, transform=tfms)

        self.loss_func = CaptionLoss(0.5, self.tokenizer)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        param_groups = [
            {'params': self.model.encoder.parameters(), 'lr': 1e-5},
            {'params': self.model.decoder.parameters(), 'lr': 1e-3}
        ]
        optimizer = torch.optim.Adam(param_groups, lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        out = self.model.encoder(x)
        loss = self.loss_func(out, y, self.model.decoder, self.device)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        out = self.model.encoder(x)
        loss = self.loss_func(out, y, self.model.decoder, self.device)

        # measuring bleu score
        score_list = []
        bs, dim = out.shape
        captions = self.model(out, self.device)

        for i, caption in enumerate(captions):
            candidate_corpus = [caption.split()]
            reference_corpus = [self.tokenizer.idx2val[j.item()] for j in y[i]]
            reference_corpus = [[reference_corpus]]

            # get the bleu score
            score = bleu_score(candidate_corpus, reference_corpus)
            score_list.append(score)

            # average the score across all item
        avg_score = np.mean(score_list)

        self.log('val_bleu_score', avg_score)
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx):
        # this is the prediction loop
        x, y = batch
        out = self.model.encoder(x)
        captions = self.model(out, self.device)
        return captions
        
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
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs)
    trainer.fit(model)