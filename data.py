import os
import shutil
import zipfile
from glob import glob
from PIL import Image
from urllib import request
import matplotlib.pyplot as plt


IMAGE_URL =  "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

files = ['Flickr8k_Dataset.zip', 'Flickr8k_text.zip']

def download_and_extract(url, fname, root):
    """Downloads the zip file from given url and extracts
    it in given folder

    Parameters
    ----------
    url : str
        url of zip file

    fname: str
        folder name where the zip file will be extracted

    Returns
    -------
    None
    """
    path = os.path.join(root, fname)
    if os.path.exists(path): # if file already exsits than dont extract it again
        print(f'File already exists at path {path}.')
        return
    print(f'Downloading file from {url} ...')
    request.urlretrieve(url, path)
    with zipfile.ZipFile(path, 'r') as f:
        print(f'extracting file at path {root} ...')
        f.extractall(root)

def download_data(root):
    download_and_extract(IMAGE_URL, files[0], root)
    download_and_extract(TEXT_URL, files[1], root)

    path = os.path.join(root, 'Flicker8k_text')
    os.makedirs(path, exist_ok=True)
    for file in glob(os.path.join(root, '*.txt')):
        fname = file.rsplit('/')[-1]
        shutil.move(file, os.path.join(path, fname))

image_files = glob('./Flicker8k_Dataset/*.jpg')

# TODO: move this funtion to differt module
def show(shape=(3, 4), figsize=(10, 10)):
    fig, axs = plt.subplots(*shape, figsize=figsize)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(Image.open(image_files[i]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()



