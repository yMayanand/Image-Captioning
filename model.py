import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone = models.resnet50()
        backbone = [module for module in backbone.children()][:-1]
        backbone.append(nn.Flatten())
        self.backbone = nn.Sequential(*backbone)
        #for param in self.backbone.parameters():
        #    param.requires_grad = False

    def forward(self, x):
        return self.backbone(x) # size (b, 512)


class Decoder(nn.Module):
    def __init__(self, tokenizer, dropout=0.5):
        super().__init__()
        self.vocab_size = len(tokenizer)
        self.emb = nn.Embedding(self.vocab_size, 512) # size (b, 512)
        self.lstm = nn.LSTMCell(512, 512) 
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, len(tokenizer.vocab))
        self.init_h = nn.Linear(2048, 512)
        self.init_c = nn.Linear(2048, 512)
    
    def init_states(self, encoder_out):
        h = self.init_h(encoder_out)
        c = self.init_c(encoder_out)
        return h, c

    def forward(self, im_hid, captions, caplens, device):
        batch_size, encoder_dim = im_hid.shape
        gen_caps = []
        caplens, sort_idx = caplens.squeeze(1).sort(dim=0, descending=True)
        sort_idx = sort_idx[::-1].tolist()
        im_hid = im_hid[sort_idx]
        captions = captions[sort_idx]
        h, c = self.init_states(im_hid)

        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)


        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        caplens = (caplens - 1).tolist()


        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(caplens), self.vocab_size).to(device)

        max_timesteps = max(caplens)

        for t in range(max_timesteps):
            batch_size_t = sum([l > t for l in caplens])
            h, c = self.lstm(embeddings[:batch_size_t, t, :], (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
        
        return  predictions, captions, caplens, sort_idx

class CaptionModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.encoder = Encoder()
        self.decoder = Decoder(tokenizer)

    def forward(self, x, captions, caplens, device):
        encoder_out = self.encoder(x)
        predictions, captions, caplens, sort_idx = self.decoder(encoder_out, captions, caplens, device)
        return predictions, captions, caplens, sort_idx