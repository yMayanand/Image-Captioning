import torch
import torch.nn as nn
from preprocess import tokenizer
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CaptionLoss(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def init_states(self):
        self.cn = torch.zeros(1, 512).to(device)
        return self.cn

    def forward(self, x, labels):
        loss = 0.
        criterion = nn.CrossEntropyLoss()
        batch_size = x.shape[0]
        for i in range(batch_size):
            inp = self.decoder.emb(torch.LongTensor([tokenizer.val2idx['START']]).to(device)) # start token
            
            #self.hn = x[i].unsqueeze(0)
            self.decoder.init_states(x[i])
            for label in labels[i]:
                label = label.unsqueeze(0).to(device)
                out, *(self.decoder.hn, self.decoder.cn) = self.decoder(inp, self.decoder.hn, self.decoder.cn)
                _, pred = torch.max(out, dim=1)
                if np.random.random() > 0.5:
                    token = torch.LongTensor([pred.item()]).to(device)
                    inp = self.decoder.emb(token)
                else:
                    inp = self.decoder.emb(label)
                loss += criterion(out, label)
            label = torch.LongTensor([tokenizer.val2idx['STOP']]).to(device) # stop token
            out, *(self.decoder.hn, self.decoder.cn) = self.decoder(inp, self.decoder.hn, self.decoder.cn)
            loss += criterion(out, label)
        return loss / batch_size