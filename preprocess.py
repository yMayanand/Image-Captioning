import os
import pickle
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

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

# function to return 2
def return2():
    return 2

class Tokenizer:
    """Tokenizer class for tokenizing captions in the Flicker8k dataset.

    Parameters
    ----------
    root : str
        root directory where dataset is stored

    """

    def __init__(self, root):
        self.vocab = ['<start>', '<end>', '<unk>', '<pad>']
        self.count = 3
        self.idx2val = {}
        self.val2idx = {'<start>': 0, '<end>': 1, '<unk>': 2, '<pad>': 3}
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
        print(f'tokenizing file {fname}...')
        temp = read_file(os.path.join(self.root, fname))
        df = pd.DataFrame(temp, columns=['id'])
        for i in df['id']:
            captions = self.caption_df[self.caption_df['id'] == i].reset_index(drop=True)['caption']
            for caption in captions:
                self.add(caption)

        self.complete()

    def complete(self):
        self.idx2val = {key: value for value, key in self.val2idx.items()}
        self.val2idx = defaultdict(return2, self.val2idx)

    def pickle_tokenizer(self, fname):
        print(f"saving to file {fname}")
        with open(fname, 'wb') as f:
            state_dict = {'idx2val': self.idx2val, 'val2idx': self.val2idx, 'vocab': self.vocab}
            pickle.dump(state_dict, f)

    def load_tokenizer(self, fname):
        print(f"loading from file {fname}...")
        with open(fname, 'rb') as f:
            state_dict = pickle.load(f)
            self.vocab = state_dict['vocab']
            self.val2idx = state_dict['val2idx']
            self.idx2val = state_dict['idx2val']

    def __len__(self):
        return len(self.vocab)
