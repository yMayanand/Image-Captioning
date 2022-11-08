import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        backbone = [module for module in backbone.children()][:-1]
        backbone.append(nn.Flatten())
        self.backbone = nn.Sequential(*backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

    
    def forward(self, x):
        return self.backbone(x) # size (b, 512)


class Decoder(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.emb = nn.Embedding(len(tokenizer.vocab), 512) # size (b, 512)
        self.lstm1 = nn.LSTMCell(512, 512) 
        self.fc = nn.Linear(512, len(tokenizer.vocab))
        self.init_h = nn.Linear(2048, 512)
        self.init_c = nn.Linear(2048, 512)
    
    def init_states(self, encoder_out):
        self.hn = self.init_h(encoder_out).unsqueeze(0)
        self.cn = self.init_c(encoder_out).unsqueeze(0)

    def forward(self, inp, hidden, cell_state):
        hidden, cell_state = self.lstm1(inp, (hidden, cell_state)) # size (b, 512)
        out = self.fc(hidden)
        return out, hidden, cell_state

class CaptionModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = Encoder()
        self.decoder = Decoder(tokenizer)

    def forward(self, x, device):
        bs, dim = x.shape
        captions = []
        for  i in range(bs):
             captions.append(self.decode_one_sample(x[i], device))
        return captions

    def decode_one_sample(self, im_hid, device):
        inp = self.decoder.emb(torch.LongTensor([self.tokenizer.val2idx['START']]).to(device))
        tot = 0
        seq = 0
        gen_caps = []
        self.decoder.init_states(im_hid)
        while True:
            out, *(self.decoder.hn, self.decoder.cn) = self.decoder(inp, self.decoder.hn, self.decoder.cn)
            _, idx = torch.max(out, dim=1)
            pred_token = self.tokenizer.idx2val[idx.item()]
            gen_caps.append(pred_token)
            tot += 1
            if (tot > 25) or (pred_token=='STOP'):
                break
            inp = self.decoder.emb(torch.LongTensor([self.tokenizer.val2idx[pred_token]]).to(device))
        return " ".join(gen_caps)