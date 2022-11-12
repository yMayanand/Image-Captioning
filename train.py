from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data.metrics import bleu_score
from torchvision import transforms

from dataset import CaptionDataset
from model import CaptionModel
from preprocess import Tokenizer
from utils import train_collate, val_collate


class Model(pl.LightningModule):
    def __init__(self, root_dir, tokenizer_path):
        super().__init__()
        train_tfms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])

        val_tfms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])

        self.tokenizer = Tokenizer(root_dir)
        if tokenizer_path is not None:
            self.tokenizer.load_tokenizer(tokenizer_path)
        else:
            self.tokenizer.tokenize('Flicker8k_text/Flickr_8k.trainImages.txt')
        

        self.model = CaptionModel(self.tokenizer)

        self.train_ds = CaptionDataset(
            root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt', self.tokenizer, transform=train_tfms)
        self.val_ds = CaptionDataset(
            root_dir, 'Flicker8k_text/Flickr_8k.devImages.txt', self.tokenizer, transform=val_tfms)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        param_groups = [
            {'params': self.model.encoder.parameters(), 'lr': 1e-5},
            {'params': self.model.decoder.parameters(), 'lr': 1e-3}
        ]
        optimizer = optim.Adam(param_groups, lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, captions, caplens = batch
        predictions, captions, caplens, sort_idx = self.model(
            x, captions, caplens, self.device)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = captions[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(predictions, caplens, batch_first=True)
        targets = pack_padded_sequence(targets, caplens, batch_first=True)

        loss = self.loss_func(scores.data, targets.data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, z = batch
        captions = self.model.predict(x, self.device)
        bleu4 = bleu_score(captions, y)

        self.log('val_bleu_score', bleu4)
        

    def predict_step(self, batch, batch_idx):
        # this is the prediction loop
        x, y = batch
        out = self.model.encoder(x)
        captions = self.model(out, self.device)
        return captions

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=32, pin_memory=True, num_workers=2, collate_fn=val_collate)
        return loader

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=train_collate)
        return loader


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", default="./",
                        type=str, help="root directory path")
    parser.add_argument("--epochs", default=10, type=int,
                        help="num of epochs to train")
    parser.add_argument("--tokenizer_path", default=None, type=str,
                        help="path to saved tokenizer")

    args = parser.parse_args()
    model = Model(args.root_dir, args.tokenizer_path)
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.epochs)
    trainer.fit(model)
