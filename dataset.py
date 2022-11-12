import os
import random

import pandas as pd
import torch
from PIL import Image

from preprocess import read_file


class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, root, fname, tokenizer, mode='train', transform=None):
        fname = os.path.join(root, fname)
        self.root = root
        self.tfms = transform
        self.mode = mode
        super().__init__()
        temp = read_file(fname)
        self.df = pd.DataFrame(temp, columns=['id'])
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['id'][idx]
        caption = self.tokenizer.caption_df[self.tokenizer.caption_df['id']==image_path].reset_index(drop=True)['caption'] # [0].lower().strip().split()
        image = Image.open(os.path.join(self.root, 'Flicker8k_Dataset', image_path))
        if self.tfms:
            image = self.tfms(image)

        if self.mode == 'train':
            captions = random.choice(caption)
            captions = captions.lower().strip().split()
            captions.insert(0, '<start>')
            captions.append('<end>')
            captions = [self.tokenizer.val2idx[i] for i in captions]
            captions = torch.LongTensor(captions)
            caplens = torch.LongTensor([len(captions)])

        else:
            captions = caption.tolist()
            for i, c in enumerate(captions):
                temp = c.lower().strip().split()
                temp.append('<end>')
                captions[i] = [self.tokenizer.val2idx[i] for i in temp]
                caplens = len(captions)

        return image, captions, caplens