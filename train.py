from io import BytesIO
from model import Encoder, Decoder
from preprocess import tokenizer, train_df, test_df
from dataset import CaptionDataset, tfms
from utils import cust_collate
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from fastapi import FastAPI, File, UploadFile
import cv2
import uvicorn
from PIL import Image
from pydantic import BaseModel



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder().to(device)
decoder = Decoder().to(device)

train_ds = CaptionDataset(train_df)
val_ds = CaptionDataset(test_df)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True, num_workers=2, collate_fn=cust_collate)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, pin_memory=True, num_workers=2, collate_fn=cust_collate)

def load_from_ckpt(encoder, decoder, path):
    weights = torch.load(path, map_location=device)
    encoder.load_state_dict(weights['encoder'])
    decoder.load_state_dict(weights['decoder'])

def train_model():  
    criterion = nn.CrossEntropyLoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-2)
    sched = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_dl), epochs=10)
    loss_store = []
    epochs = 10

    for i in tqdm(range(epochs)):
        temp_store = []
        for data, labels in tqdm(train_dl, total=len(train_dl), leave=False):
            encoder.train()
            decoder.train()
            data = data.to(device)
            labels = labels
            batch_size = len(labels)
            im_hid = encoder(data)
            losses = []
            for i in range(batch_size):
                label = labels[i]
                seq_len = label.shape[0]
                inp = decoder.emb(torch.LongTensor([tokenizer.val2idx['START']]).to(device)) # start token
                hid_state = im_hid[i].unsqueeze(0)
                cell_state = torch.zeros_like(inp)
                for j in range(seq_len):
                    per_seq = label[j].unsqueeze(0).to(device)
                    out, hid_state, cell_state = decoder(inp, hid_state, cell_state)
                    losses.append(criterion(out, per_seq))
                    if np.random.random() > 0.7:
                        _, idx = torch.max(out.detach(), dim=1)
                        temp = torch.LongTensor([idx.item()]).to(device)
                        inp = decoder.emb(temp)
                    else:
                        inp = decoder.emb(per_seq)
                end = torch.LongTensor([tokenizer.val2idx['STOP']]).to(device) # stop token
                out, hid_state, cell_state = decoder(inp, hid_state, cell_state)
                losses.append(criterion(out, end))
            final_loss = torch.stack(losses)
            final_loss = torch.sum(final_loss)
            final_loss.backward()
            optimizer.step()
            sched.step()
            optimizer.zero_grad()
            temp_store.append(final_loss.item())
        print(f"Loss: {np.mean(temp_store)}")
        loss_store += temp_store

load_from_ckpt(encoder, decoder, './checkpoint/caption.pt')

def predict(data):
    encoder.eval()
    decoder.eval()
    data = data.to(device)
    im_hid = encoder(data)
    inp = decoder.emb(torch.LongTensor([tokenizer.val2idx['START']]).to(device))
    tot = 0
    seq = 0
    gen_caps = []
    decoder.init_states(im_hid[0])
    while True:
        out, *(decoder.hn, decoder.cn) = decoder(inp, decoder.hn, decoder.cn)
        _, idx = torch.max(out, dim=1)
        pred_token = tokenizer.idx2val[idx.item()]
        gen_caps.append(pred_token)
        tot += 1
        if (tot > 25) or (pred_token=='STOP'):
            break
        inp = decoder.emb(torch.LongTensor([tokenizer.val2idx[pred_token]]).to(device))
    return " ".join(gen_caps)

def read_image(image):
    print(type(image))
    image = Image.open(BytesIO(image))
    return image

app = FastAPI(title='Image Captioning')



def predict_from_file(file):
    image = read_image(file)
    image = np.asarray(image.resize((256, 256)))
    prediction = predict(tfms(image).unsqueeze(0)) 
    return {'preds': prediction}


@app.post("/predict/")
async def predict_api(file:  UploadFile): 
    print(file)
    content = await file.read() 
    
    return predict_from_file(content)
