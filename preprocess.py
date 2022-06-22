import pandas as pd
from pathlib import Path
from collections import defaultdict

def parse_info(line):
    img_name, *caption = line.strip().split()
    return img_name, ' '.join(caption)

def cap_iter():
    with open('./Flicker8k_text/Flickr8k.token.txt', 'r') as f:
        for line in f:
                yield parse_info(line)


with open('./Flicker8k_text/Flickr8k.token.txt', 'r') as f:
    lines = []
    for line in f:
        line = line.strip().split()
        line = [line[0], " ".join(line[1:])]
        lines.append(line)

caption_df = pd.DataFrame(lines, columns=['id', 'caption'])
caption_df[['id', 'caption']]

caption_df['id'], caption_df['cap_no'] = caption_df['id'].str.split('#').apply(lambda x: x[0]), caption_df['id'].str.split('#').apply(lambda x: x[1])
path = Path('./Flicker8k_Dataset')

def read_file(fname):
    with open(fname, 'r') as f:
        lines = []
        for line in f:
            lines.append(line.strip())
    return lines


class Tokenize:
    def __init__(self):
        self.vocab = ['START', 'STOP', 'UNK']
        self.count = 2
        self.idx2val = {}
        self.val2idx = {'START': 1, 'STOP': 2, 'UNK': 0}

    def add(self, text):
        for i in text.lower().strip().split():
            if i not in self.val2idx.keys():
                self.count += 1
                self.vocab.append(i)
                self.val2idx.update({i: self.count})
    
    def complete(self):
        self.idx2val = {key: value for value, key in self.val2idx.items()}
        self.val2idx = defaultdict(int, self.val2idx)

temp = read_file('./Flicker8k_text/Flickr_8k.trainImages.txt')
train_df = pd.DataFrame(temp, columns=['id'])
temp = read_file('./Flicker8k_text/Flickr_8k.devImages.txt')
val_df = pd.DataFrame(temp, columns=['id'])
temp = read_file('./Flicker8k_text/Flickr_8k.testImages.txt')
test_df = pd.DataFrame(temp, columns=['id'])

tokenizer = Tokenize()
for i in train_df['id']:
    caption = caption_df[caption_df['id']==i].reset_index(drop=True)['caption'][0]
    tokenizer.add(caption)
tokenizer.complete()

    