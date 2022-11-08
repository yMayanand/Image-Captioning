import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CaptionLoss(nn.Module):
    def __init__(self, teacher_forcing, tokenizer):
        super().__init__()
        self.teacher_forcing = teacher_forcing
        self.tokenizer = tokenizer

    def forward(self, x, labels, decoder):
        loss = 0.
        criterion = nn.CrossEntropyLoss()
        batch_size = x.shape[0]
        for i in range(batch_size):
            inp = decoder.emb(torch.LongTensor([self.tokenizer.val2idx['START']]).to(device)) # start token
            
            decoder.init_states(x[i])
            for label in labels[i]:
                label = label.unsqueeze(0).to(device)
                out, *(decoder.hn, decoder.cn) = decoder(inp, decoder.hn, decoder.cn)
                _, pred = torch.max(out, dim=1)
                if np.random.random() > self.teacher_forcing:
                    token = torch.LongTensor([pred.item()]).to(device)
                    inp = decoder.emb(token)
                else:
                    inp = decoder.emb(label)
                loss += criterion(out, label)
            label = torch.LongTensor([self.tokenizer.val2idx['STOP']]).to(device) # stop token
            out, *(decoder.hn, decoder.cn) = decoder(inp, decoder.hn, decoder.cn)
            loss += criterion(out, label)
        return loss / batch_size