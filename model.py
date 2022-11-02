from torchvision import models
import torch.nn as nn

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