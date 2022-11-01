import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile

def read_image(image):
    image = Image.open(BytesIO(image))
    return image

app = FastAPI(title='Image Captioning')

def predict_from_file(file):
    image = read_image(file)
    image = np.asarray(image.resize((256, 256)))
    prediction = predict(tfms(image).unsqueeze(0)) 
    return {'preds': prediction}

@app.post("/predict/")
async def predict_api(file: UploadFile): 
    content = await file.read() 
    return predict_from_file(content)
