import torch
from torchvision import transforms
from preprocess import tokenizer, caption_df
from PIL import Image
import os

path = './Flicker8k_Dataset'

tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

class CaptionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df['id'][idx]
        caption = caption_df[caption_df['id']==image_path].reset_index(drop=True)['caption'][0].lower().strip().split()
        caption = [tokenizer.val2idx[i] for i in caption]
        image = Image.open(os.path.join(path, image_path))
        image = tfms(image)
        return image, torch.tensor(caption)