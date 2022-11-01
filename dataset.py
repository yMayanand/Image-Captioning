import os
import torch
from PIL import Image
import pandas as pd
from preprocess import tokenizer, read_file

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, fname, transform=None):
        self.root = fname.split('/')[0]
        self.tfms = transform
        super().__init__()
        temp = read_file(fname)
        self.df = pd.DataFrame(temp, columns=['id'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['id'][idx]
        caption = tokenizer.caption_df[tokenizer.caption_df['id']==image_path].reset_index(drop=True)['caption'][0].lower().strip().split()
        caption = [tokenizer.val2idx[i] for i in caption]
        image = Image.open(os.path.join(self.root, 'Flicker8k_Dataset', image_path))
        if self.tfms:
            image = self.tfms(image)
        return image, torch.tensor(caption)