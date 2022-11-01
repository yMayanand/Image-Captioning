import os
import pandas as pd
from pathlib import Path
from collections import defaultdict

# TODO: Remove this function


def parse_info(line):
    img_name, *caption = line.strip().split()
    return img_name, ' '.join(caption)

# TODO: Remove this function


def cap_iter():
    with open('./Flicker8k_text/Flickr8k.token.txt', 'r') as f:
        for line in f:
            yield parse_info(line)


def read_file(fname):
    with open(fname, 'r') as f:
        lines = []
        for line in f:
            lines.append(line.strip())
    return lines


class Tokenizer:
    """Tokenizer class for tokenizing captions in the Flicker8k dataset.

    Parameters
    ----------
    root : str
        root directory where dataset is stored

    """

    def __init__(self, root):
        self.vocab = ['START', 'STOP', 'UNK']
        self.count = 2
        self.idx2val = {}
        self.val2idx = {'START': 0, 'STOP': 1, 'UNK': 2}
        self.root = root

        with open(os.path.join(root, 'Flicker8k_text/Flickr8k.token.txt'), 'r') as f:
            lines = []
            for line in f:
                line = line.strip().split()
                line = [line[0], " ".join(line[1:])]
                lines.append(line)

        self.caption_df = pd.DataFrame(lines, columns=['id', 'caption'])
        self.caption_df['id'], self.caption_df['cap_no'] = self.caption_df['id'].str.split('#').apply(lambda x: x[0]), \
            self.caption_df['id'].str.split('#').apply(lambda x: x[1])

    def add(self, text):
        for i in text.lower().strip().split():
            if i not in self.val2idx.keys():
                self.count += 1
                self.vocab.append(i)
                self.val2idx.update({i: self.count})

    def tokenize(self, fname):
        temp = read_file(os.path.join(self.root, fname))
        df = pd.DataFrame(temp, columns=['id'])
        for i in df['id']:
            caption = self.caption_df[self.caption_df['id'] == i].reset_index(drop=True)['caption'][0]
            self.add(caption)
        self.complete()

    def complete(self):
        self.idx2val = {key: value for value, key in self.val2idx.items()}
        self.val2idx = defaultdict(lambda: 2, self.val2idx)



temp = read_file('./Flicker8k_text/Flickr_8k.devImages.txt')
val_df = pd.DataFrame(temp, columns=['id'])
temp = read_file('./Flicker8k_text/Flickr_8k.testImages.txt')
test_df = pd.DataFrame(temp, columns=['id'])
