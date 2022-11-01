import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CaptionLoss(nn.Module):
    def __init__(self, decoder, teacher_forcing, tokenizer):
        super().__init__()
        self.decoder = decoder
        self.teacher_forcing = teacher_forcing
        self.tokenizer = tokenizer

    def forward(self, x, labels):
        loss = 0.
        criterion = nn.CrossEntropyLoss()
        batch_size = x.shape[0]
        for i in range(batch_size):
            inp = self.decoder.emb(torch.LongTensor([tokenizer.val2idx['START']]).to(device)) # start token
            
            self.decoder.init_states(x[i])
            for label in labels[i]:
                label = label.unsqueeze(0).to(device)
                out, *(self.decoder.hn, self.decoder.cn) = self.decoder(inp, self.decoder.hn, self.decoder.cn)
                _, pred = torch.max(out, dim=1)
                if np.random.random() > self.teacher_forcing:
                    token = torch.LongTensor([pred.item()]).to(device)
                    inp = self.decoder.emb(token)
                else:
                    inp = self.decoder.emb(label)
                loss += criterion(out, label)
            label = torch.LongTensor([self.tokenizer.val2idx['STOP']]).to(device) # stop token
            out, *(self.decoder.hn, self.decoder.cn) = self.decoder(inp, self.decoder.hn, self.decoder.cn)
            loss += criterion(out, label)
        return loss / batch_size