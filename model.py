import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone = [module for module in backbone.children()][:-1]
        backbone.append(nn.Flatten())
        self.backbone = nn.Sequential(*backbone)
     

    def forward(self, x):
        return self.backbone(x)
    
    def fine_tune(self, fine_tune=False):
        for param in self.parameters():
            param.requires_grad = fine_tune


class Decoder(nn.Module):
    def __init__(self, tokenizer, teacher_forcing=0.5, dropout=0.5):
        super().__init__()
        self.tokenizer = tokenizer
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

    def forward(self, enc_out, captions, caplens, device):
        batch_size = enc_out.shape[0]
        caplens, sort_idx = caplens.squeeze(1).sort(dim=0, descending=True)
        enc_out = enc_out[sort_idx]
        captions = captions[sort_idx]
        h, c = self.init_states(enc_out)

        # Embedding
        embeddings = self.emb(captions)  # (batch_size, max_caption_length, embed_dim)


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

    def predict(self, enc_out, device, max_steps):
        with torch.no_grad():
            batch_size = enc_out.shape[0]
            h, c = self.init_states(enc_out)

            captions = []

            for i in range(batch_size):
                temp = []
                next_token = self.emb(torch.LongTensor([self.tokenizer.val2idx['<start>']]).to(device))
                h_, c_ = h[i].unsqueeze(0), c[i].unsqueeze(0)

                step = 1
                while True:
                    h_, c_ = self.lstm(next_token, (h_, c_))
                    preds = self.fc(self.dropout(h_))

                    max_val, max_idx = torch.max(preds, dim=1)
                    max_idx = max_idx.item()
                    temp.append(max_idx)
                    
                    if max_idx in [self.tokenizer.val2idx['<end>']] or step == max_steps:
                        break
                    next_token = self.emb(torch.LongTensor([max_idx]).to(device))
                    step += 1
                captions.append(temp)
        return  captions

    

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
    
    def predict(self, x, device, max_steps=25):
        encoder_out = self.encoder(x)
        captions = self.decoder.predict(encoder_out, device, max_steps)
        return captions