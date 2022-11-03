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

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

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

def ddp_setup():
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        root_dir,
        lr,
        save_every,
        snapshot_path,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        tfms = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()
        ])

        tokenizer = Tokenizer(root_dir)
        tokenizer.tokenize(os.path.join(root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt'))

        self.encoder = Encoder().to(device)
        self.decoder = Decoder(tokenizer).to(device)

        self.decoder = DDP(self.decoder, device_ids=[self.local_rank])

        params = list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.criterion = CaptionLoss(self.decoder, 0.5, tokenizer)

        train_ds = CaptionDataset(root_dir, 'Flicker8k_text/Flickr_8k.trainImages.txt', tokenizer, transform=tfms)
        val_ds = CaptionDataset(root_dir, 'Flicker8k_text/Flickr_8k.devImages.txt', tokenizer, transform=tfms)

        self.train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset), num_workers=2, collate_fn=cust_collate)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, pin_memory=True, num_workers=2, collate_fn=cust_collate)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.encoder(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_dl))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_dl.sampler.set_epoch(epoch)
        loss_store = []
        for source, targets in self.train_dl:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            loss = self._run_batch(source, targets)
            loss_store.append(loss)
        avg_loss = np.mean(loss_store)
        print(f"Epoch {epoch} Loss: {avg_loss}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)



def main(root_dir, lr, save_every, total_epochs, snapshot_path):
    ddp_setup()
    trainer = Trainer(root_dir, lr, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", default="./", type=str, help="root directory path")
    parser.add_argument("--epochs", default=10, type=int, help="num of epochs to train")
    parser.add_argument("--lr", default=1e-2, type=float, help="learing rate")
    parser.add_argument("--save_every", default=2, type=int, help="saving snapshot of model every n epochs")
    parser.add_argument("--snapshot_path", default='snapshot.pt', type=str, help="path to save snapshot")



    args = parser.parse_args()

    main(args.root_dir, args.lr, args.save_every, args.epochs, args.snapshot_path)

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
