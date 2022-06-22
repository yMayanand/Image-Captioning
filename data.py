from urllib import request
import zipfile
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

IMAGE_URL =  "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

files = ['Flickr8k_Dataset.zip', 'Flickr8k_text.zip']

def download_and_extract(url, fname):
    path = os.path.join('./', fname)
    if os.path.exists(path):
        return
    request.urlretrieve(url, fname)
    with zipfile.ZipFile(fname, 'r') as f:
        f.extractall()

download_and_extract(IMAGE_URL, files[0])
download_and_extract(TEXT_URL, files[1])

path = './Flicker8k_text'
os.makedirs(path, exist_ok=True)
for file in glob('./*.txt'):
    fname = file.rsplit('/')[-1]
    shutil.move(file, os.path.join(path, fname))

image_files = glob('./Flicker8k_Dataset/*.jpg')

def show(shape=(3, 4), figsize=(10, 10)):
    fig, axs = plt.subplots(*shape, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(Image.open(image_files[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

show()

