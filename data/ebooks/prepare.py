import os, glob, tqdm
import requests
import tiktoken
import numpy as np

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from bs4 import BeautifulSoup
from utillc import *

blacklist = [   '[document]',   'noscript', 'header',   'html', 'meta', 'head','input', 'script',   ]

# download the tiny shakespeare dataset
input_dir = "/mnt/NUC/data/books"
input_dir = "./bb"
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

def thtml2ttext(thtml):
    Output = []
    for html in thtml:
        text =  chap2text(html)
        Output.append(text)
    return Output

def chap2text(chap):
    output = ''
    soup = BeautifulSoup(chap, 'html.parser')
    text = soup.find_all(text=True)
    for t in text:
        if t.parent.name not in blacklist:
            output += '{} '.format(t)
    return output.replace("\xa0","")

def get_text(path) :
    book = epub.read_epub(path)
    chapters = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            #EKOX(item.get_content())
            chapters.append(item.get_content())
        return chapters

def text() :
    r = ""
    files = glob.glob(input_dir + '/*.epub')
    EKOX(len(files))
    for f in tqdm.tqdm(files[0:20]) :
        try :
            EKOX(f)
            chapters = get_text(f)
            ttext = thtml2ttext(chapters)
            r += " " + ttext
        except Exception as e:
            EKOX(e)
            #raise(e)
            pass
    return r

data = text()

n = len(data)
EKOX(n)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
